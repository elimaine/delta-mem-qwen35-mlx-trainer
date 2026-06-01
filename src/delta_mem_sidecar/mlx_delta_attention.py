from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from delta_mem_sidecar.mlx_delta_math import delta_affine_scan


@dataclass(frozen=True)
class MlxDeltaAttentionWeights:
    memory_q_proj: Any
    memory_k_proj: Any
    memory_v_proj: Any
    delta_q_proj: Any
    delta_o_proj: Any
    beta_proj: Any
    beta_bias: Any
    lambda_proj: Any | None = None
    lambda_bias: Any | None = None


@dataclass(frozen=True)
class MlxDeltaOutputWeights:
    memory_q_proj: Any
    memory_k_proj: Any
    memory_v_proj: Any
    delta_o_proj: Any
    beta_proj: Any
    beta_bias: Any
    lambda_proj: Any | None = None
    lambda_bias: Any | None = None


@dataclass(frozen=True)
class MlxDeltaAttentionConfig:
    rank: int
    alpha: float = 1.0
    q_delta_scale: float = 1.0
    o_delta_scale: float = 1.0
    max_delta_ratio: float | None = None
    active_delta_heads: frozenset[str] = frozenset({"q", "o"})
    normalize_qk: bool = True
    couple_lambda: bool = True
    state_update_mode: str = "standard"

    @property
    def delta_scaling(self) -> float:
        return self.alpha / self.rank


@dataclass(frozen=True)
class MlxAttentionShape:
    n_heads: int
    n_kv_heads: int
    head_dim: int
    query_out_dims: int
    q_proj_out_dims: int | None
    is_query_gated: bool


class MlxDeltaAttention:
    """MLX wrapper that applies δ-mem Q/O corrections around Qwen attention."""

    def __init__(
        self,
        base_attention: Any,
        weights: MlxDeltaAttentionWeights,
        config: MlxDeltaAttentionConfig,
    ) -> None:
        self.base = base_attention
        self.weights = weights
        self.delta_config = config
        shape = infer_mlx_attention_shape(base_attention, weights)
        self.attention_shape = shape
        self.n_heads = shape.n_heads
        self.n_kv_heads = shape.n_kv_heads
        self.head_dim = shape.head_dim
        self.query_out_dims = shape.query_out_dims
        self.is_query_gated = shape.is_query_gated
        self.scale = base_attention.scale
        self.q_proj = base_attention.q_proj
        self.k_proj = base_attention.k_proj
        self.v_proj = base_attention.v_proj
        self.o_proj = base_attention.o_proj
        self.rope = base_attention.rope
        self.q_norm = getattr(base_attention, "q_norm", None)
        self.k_norm = getattr(base_attention, "k_norm", None)
        self.delta_state: Any | None = None
        self.write_enabled = True
        self.last_reads: Any | None = None
        self.last_query_gate: Any | None = None

    def reset_state(self) -> None:
        self.delta_state = None
        self.last_reads = None

    def state_snapshot(self) -> Any | None:
        return self.delta_state

    def load_state_snapshot(self, state: Any | None) -> None:
        self.delta_state = state

    def set_write_enabled(self, enabled: bool) -> None:
        self.write_enabled = enabled

    def __call__(
        self,
        x: Any,
        mask: Any | None = None,
        cache: Any | None = None,
    ) -> Any:
        import mlx.core as mx

        batch_size, seq_len, _ = x.shape
        memory_q, memory_k, memory_v, beta, lam = self._memory_sequence_projections(x)
        state = self._ensure_state(batch_size, memory_q.dtype)
        if self.write_enabled:
            keep, erase, write = self._memory_update_coefficients(beta, lam)
            state, reads = delta_affine_scan(
                state,
                memory_q,
                memory_k,
                memory_v,
                keep,
                erase,
                write,
            )
            self.delta_state = state
        else:
            reads = mx.matmul(state, mx.expand_dims(memory_q, -1)).squeeze(-1)
        self.last_reads = reads

        queries, query_gate = self._query_projection_with_optional_gate(x)
        delta_q = self._project_delta(reads, self.weights.delta_q_proj, "q")
        if delta_q is not None:
            if tuple(delta_q.shape) != tuple(queries.shape):
                raise RuntimeError(
                    "δ-mem q delta shape mismatch: "
                    f"delta has {tuple(delta_q.shape)}, query has {tuple(queries.shape)}"
                )
            delta_q = _limit_delta_by_rms(delta_q, queries, self.delta_config.max_delta_ratio)
            queries = queries + delta_q.astype(queries.dtype)
        self.last_query_gate = query_gate
        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = queries.reshape(batch_size, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = keys.reshape(batch_size, seq_len, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(batch_size, seq_len, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        if self.q_norm is not None:
            queries = self.q_norm(queries)
        if self.k_norm is not None:
            keys = self.k_norm(keys)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = _scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        if self.is_query_gated:
            if query_gate is None:
                raise RuntimeError("Qwen3.5 query-gated attention did not capture a query gate")
            output = output * mx.sigmoid(query_gate).astype(output.dtype)
        base_output = self.o_proj(output)
        delta_o = self._project_delta(reads, self.weights.delta_o_proj, "o")
        if delta_o is not None:
            delta_o = _limit_delta_by_rms(delta_o, base_output, self.delta_config.max_delta_ratio)
            base_output = base_output + delta_o.astype(base_output.dtype)
        return base_output

    def _ensure_state(self, batch_size: int, dtype: Any) -> Any:
        import mlx.core as mx

        rank = self.delta_config.rank
        if self.delta_state is None or tuple(self.delta_state.shape) != (batch_size, rank, rank):
            self.delta_state = mx.zeros((batch_size, rank, rank), dtype=dtype)
        return self.delta_state

    def _memory_sequence_projections(self, x: Any) -> tuple[Any, Any, Any, Any, Any]:
        import mlx.core as mx

        memory_q = _linear(x, self.weights.memory_q_proj)
        memory_k = _linear(x, self.weights.memory_k_proj)
        memory_v = _linear(x, self.weights.memory_v_proj)
        if self.delta_config.normalize_qk:
            memory_q = _normalize_memory_projection(memory_q)
            memory_k = _normalize_memory_projection(memory_k)

        beta = mx.sigmoid(_linear(x, self.weights.beta_proj) + self.weights.beta_bias)
        if self.delta_config.state_update_mode == "no_lambda":
            lam = mx.ones_like(beta)
        elif self.delta_config.couple_lambda:
            lam = 1.0 - beta
        else:
            if self.weights.lambda_proj is None or self.weights.lambda_bias is None:
                raise ValueError("lambda weights are required when couple_lambda=False")
            lam = mx.sigmoid(_linear(x, self.weights.lambda_proj) + self.weights.lambda_bias)
        return memory_q, memory_k, memory_v, beta, lam

    def _project_delta(self, reads: Any, weight: Any, head_name: str) -> Any | None:
        if head_name not in self.delta_config.active_delta_heads:
            return None
        head_scale = self.delta_config.q_delta_scale if head_name == "q" else self.delta_config.o_delta_scale
        return _linear(reads, weight) * self.delta_config.delta_scaling * head_scale

    def _query_projection_with_optional_gate(self, x: Any) -> tuple[Any, Any | None]:
        projected = self.q_proj(x)
        if not self.is_query_gated:
            return projected, None

        import mlx.core as mx

        batch_size, seq_len, _ = projected.shape
        projected = projected.reshape(batch_size, seq_len, self.n_heads, self.head_dim * 2)
        query, gate = mx.split(projected, 2, axis=-1)
        query = query.reshape(batch_size, seq_len, self.query_out_dims)
        gate = gate.reshape(batch_size, seq_len, self.query_out_dims)
        return query, gate

    def _memory_update_coefficients(self, beta: Any, lam: Any) -> tuple[Any, Any, Any]:
        import mlx.core as mx

        if self.delta_config.state_update_mode == "standard":
            return lam, beta, beta
        if self.delta_config.state_update_mode == "lambda_outside":
            return lam, lam * beta, beta
        if self.delta_config.state_update_mode == "no_lambda":
            return mx.ones_like(beta), beta, beta
        raise ValueError(f"Unsupported state_update_mode: {self.delta_config.state_update_mode}")


class MlxDeltaGatedDeltaNetOutput:
    """Output-only δ-mem wrapper for Qwen3.5 GatedDeltaNet layers."""

    def __init__(
        self,
        base_linear_attention: Any,
        weights: MlxDeltaOutputWeights,
        config: MlxDeltaAttentionConfig,
    ) -> None:
        self.base = base_linear_attention
        self.weights = weights
        self.delta_config = config
        self.hidden_size = getattr(base_linear_attention, "hidden_size")
        self.delta_state: Any | None = None
        self.write_enabled = True
        self.last_reads: Any | None = None

    def reset_state(self) -> None:
        self.delta_state = None
        self.last_reads = None

    def state_snapshot(self) -> Any | None:
        return self.delta_state

    def load_state_snapshot(self, state: Any | None) -> None:
        self.delta_state = state

    def set_write_enabled(self, enabled: bool) -> None:
        self.write_enabled = enabled

    def __call__(self, inputs: Any, mask: Any | None = None, cache: Any | None = None) -> Any:
        import mlx.core as mx

        base_output = self.base(inputs, mask=mask, cache=cache)
        memory_q, memory_k, memory_v, beta, lam = _memory_sequence_projections(
            inputs,
            weights=self.weights,
            config=self.delta_config,
        )
        state = self._ensure_state(int(inputs.shape[0]), memory_q.dtype)
        if self.write_enabled:
            keep, erase, write = _memory_update_coefficients(beta, lam, self.delta_config.state_update_mode)
            state, reads = delta_affine_scan(state, memory_q, memory_k, memory_v, keep, erase, write)
            self.delta_state = state
        else:
            reads = mx.matmul(state, mx.expand_dims(memory_q, -1)).squeeze(-1)
        self.last_reads = reads

        delta_o = _linear(reads, self.weights.delta_o_proj) * self.delta_config.delta_scaling * self.delta_config.o_delta_scale
        delta_o = _limit_delta_by_rms(delta_o, base_output, self.delta_config.max_delta_ratio)
        return base_output + delta_o.astype(base_output.dtype)

    def _ensure_state(self, batch_size: int, dtype: Any) -> Any:
        import mlx.core as mx

        rank = self.delta_config.rank
        if self.delta_state is None or tuple(self.delta_state.shape) != (batch_size, rank, rank):
            self.delta_state = mx.zeros((batch_size, rank, rank), dtype=dtype)
        return self.delta_state


def wrap_qwen_attention_layers(
    model: Any,
    *,
    weights_by_layer: dict[int, MlxDeltaAttentionWeights],
    config: MlxDeltaAttentionConfig,
) -> list[int]:
    wrapped: list[int] = []
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers")
    for layer_index, weights in weights_by_layer.items():
        layer = layers[layer_index]
        layer.self_attn = MlxDeltaAttention(layer.self_attn, weights, config)
        wrapped.append(layer_index)
    return wrapped


def wrap_qwen_gated_delta_output_layers(
    model: Any,
    *,
    weights_by_layer: dict[int, MlxDeltaOutputWeights],
    config: MlxDeltaAttentionConfig,
) -> list[int]:
    wrapped: list[int] = []
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers")
    for layer_index, weights in weights_by_layer.items():
        layer = layers[layer_index]
        if not getattr(layer, "is_linear", False) or not hasattr(layer, "linear_attn"):
            raise RuntimeError(f"layer {layer_index} is not a Qwen3.5 GatedDeltaNet layer")
        layer.linear_attn = MlxDeltaGatedDeltaNetOutput(layer.linear_attn, weights, config)
        wrapped.append(layer_index)
    return wrapped


def iter_mlx_delta_attention_modules(model: Any):
    inner = getattr(model, "model", model)
    for layer_index, layer in enumerate(getattr(inner, "layers", [])):
        attention = getattr(layer, "self_attn", None)
        if isinstance(attention, MlxDeltaAttention):
            yield layer_index, attention


def iter_mlx_delta_output_modules(model: Any):
    inner = getattr(model, "model", model)
    for layer_index, layer in enumerate(getattr(inner, "layers", [])):
        attention = getattr(layer, "linear_attn", None)
        if isinstance(attention, MlxDeltaGatedDeltaNetOutput):
            yield layer_index, attention


def reset_mlx_delta_states(model: Any) -> None:
    for _, attention in iter_mlx_delta_attention_modules(model):
        attention.reset_state()
    for _, attention in iter_mlx_delta_output_modules(model):
        attention.reset_state()


def set_mlx_delta_write_enabled(model: Any, enabled: bool) -> None:
    for _, attention in iter_mlx_delta_attention_modules(model):
        attention.set_write_enabled(enabled)
    for _, attention in iter_mlx_delta_output_modules(model):
        attention.set_write_enabled(enabled)


def get_mlx_delta_state(model: Any) -> dict[int, Any]:
    state = {
        layer_index: attention.state_snapshot()
        for layer_index, attention in iter_mlx_delta_attention_modules(model)
        if attention.state_snapshot() is not None
    }
    state.update(
        {
            layer_index: attention.state_snapshot()
            for layer_index, attention in iter_mlx_delta_output_modules(model)
            if attention.state_snapshot() is not None
        }
    )
    return state


def load_mlx_delta_state(model: Any, state: dict[int, Any]) -> None:
    for layer_index, attention in iter_mlx_delta_attention_modules(model):
        attention.load_state_snapshot(state.get(layer_index))
    for layer_index, attention in iter_mlx_delta_output_modules(model):
        attention.load_state_snapshot(state.get(layer_index))


def infer_mlx_attention_shape(
    base_attention: Any,
    weights: MlxDeltaAttentionWeights,
) -> MlxAttentionShape:
    query_out_dims = int(weights.delta_q_proj.shape[0])
    q_proj_out_dims = _linear_output_dims(getattr(base_attention, "q_proj", None))
    head_dim = getattr(base_attention, "head_dim", None)
    n_heads = (
        getattr(base_attention, "n_heads", None)
        or getattr(base_attention, "num_attention_heads", None)
        or _config_attr(base_attention, "num_attention_heads")
    )
    n_kv_heads = (
        getattr(base_attention, "n_kv_heads", None)
        or getattr(base_attention, "num_key_value_heads", None)
        or _config_attr(base_attention, "num_key_value_heads")
    )

    likely_query_gated = _is_likely_query_gated_attention(base_attention)
    is_query_gated = q_proj_out_dims == query_out_dims * 2
    if likely_query_gated and q_proj_out_dims == query_out_dims:
        raise RuntimeError(
            "δ-mem adapter q output mismatch: Qwen3.5 query-gated attention "
            f"expects effective query width {q_proj_out_dims // 2}, got {query_out_dims}"
        )
    if head_dim is None and n_heads is not None:
        head_dim = query_out_dims // int(n_heads)
    if n_heads is None and head_dim is not None:
        n_heads = query_out_dims // int(head_dim)
    if n_kv_heads is None:
        n_kv_heads = n_heads

    if n_heads is None or head_dim is None or n_kv_heads is None:
        raise RuntimeError("Could not infer MLX attention head shape for δ-mem wrapper")
    if query_out_dims != int(n_heads) * int(head_dim):
        raise RuntimeError(
            "δ-mem adapter q output is incompatible with attention heads: "
            f"query width {query_out_dims}, n_heads {n_heads}, head_dim {head_dim}"
        )
    if q_proj_out_dims is not None and q_proj_out_dims not in {query_out_dims, query_out_dims * 2}:
        raise RuntimeError(
            "δ-mem adapter q output mismatch: "
            f"adapter effective query width {query_out_dims}, model q projection output {q_proj_out_dims}"
        )

    return MlxAttentionShape(
        n_heads=int(n_heads),
        n_kv_heads=int(n_kv_heads),
        head_dim=int(head_dim),
        query_out_dims=query_out_dims,
        q_proj_out_dims=q_proj_out_dims,
        is_query_gated=is_query_gated,
    )


def _linear_output_dims(linear: Any | None) -> int | None:
    if linear is None:
        return None
    output_dims = getattr(linear, "output_dims", None)
    if output_dims is not None:
        return int(output_dims)
    parameters = getattr(linear, "parameters", None)
    if callable(parameters):
        try:
            weight = parameters().get("weight")
        except Exception:  # noqa: BLE001
            weight = None
        if weight is not None and hasattr(weight, "shape") and len(weight.shape) >= 1:
            return int(weight.shape[0])
    weight = getattr(linear, "weight", None)
    if weight is not None and hasattr(weight, "shape") and len(weight.shape) >= 1:
        return int(weight.shape[0])
    return None


def _linear_input_dims(linear: Any | None) -> int | None:
    if linear is None:
        return None
    input_dims = getattr(linear, "input_dims", None)
    if input_dims is not None:
        return int(input_dims)
    return None


def _config_attr(module: Any, name: str) -> Any | None:
    config = getattr(module, "config", None) or getattr(module, "args", None)
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get(name)
    return getattr(config, name, None)


def _is_likely_query_gated_attention(module: Any) -> bool:
    names = [
        type(module).__name__.lower(),
        str(getattr(_config_attr(module, "model_type"), "lower", lambda: "")()),
        str(getattr(module, "model_type", "")).lower(),
    ]
    return any("qwen35" in name or "qwen3_5" in name or "qwen3next" in name for name in names)


def _linear(x: Any, weight: Any) -> Any:
    return x @ weight.T


def _scaled_dot_product_attention(*args: Any, **kwargs: Any) -> Any:
    from mlx_lm.models.base import scaled_dot_product_attention

    return scaled_dot_product_attention(*args, **kwargs)


def _memory_sequence_projections(
    x: Any,
    *,
    weights: MlxDeltaOutputWeights,
    config: MlxDeltaAttentionConfig,
) -> tuple[Any, Any, Any, Any, Any]:
    import mlx.core as mx

    memory_q = _linear(x, weights.memory_q_proj)
    memory_k = _linear(x, weights.memory_k_proj)
    memory_v = _linear(x, weights.memory_v_proj)
    if config.normalize_qk:
        memory_q = _normalize_memory_projection(memory_q)
        memory_k = _normalize_memory_projection(memory_k)

    beta = mx.sigmoid(_linear(x, weights.beta_proj) + weights.beta_bias)
    if config.state_update_mode == "no_lambda":
        lam = mx.ones_like(beta)
    elif config.couple_lambda:
        lam = 1.0 - beta
    else:
        if weights.lambda_proj is None or weights.lambda_bias is None:
            raise ValueError("lambda weights are required when couple_lambda=False")
        lam = mx.sigmoid(_linear(x, weights.lambda_proj) + weights.lambda_bias)
    return memory_q, memory_k, memory_v, beta, lam


def _memory_update_coefficients(beta: Any, lam: Any, state_update_mode: str) -> tuple[Any, Any, Any]:
    import mlx.core as mx

    if state_update_mode == "standard":
        return lam, beta, beta
    if state_update_mode == "lambda_outside":
        return lam, lam * beta, beta
    if state_update_mode == "no_lambda":
        return mx.ones_like(beta), beta, beta
    raise ValueError(f"Unsupported state_update_mode: {state_update_mode}")


def _limit_delta_by_rms(delta: Any, base: Any, max_ratio: float | None) -> Any:
    if max_ratio is None or max_ratio <= 0:
        return delta

    import mlx.core as mx

    work_dtype = mx.float32
    delta_f = delta.astype(work_dtype)
    base_f = base.astype(work_dtype)
    eps = mx.array(1e-6, dtype=work_dtype)
    delta_rms = mx.sqrt(mx.mean(mx.square(delta_f), axis=-1, keepdims=True) + eps)
    base_rms = mx.sqrt(mx.mean(mx.square(base_f), axis=-1, keepdims=True) + eps)
    scale = mx.minimum(mx.ones_like(delta_rms), (float(max_ratio) * base_rms) / delta_rms)
    return (delta_f * scale).astype(delta.dtype)


def _normalize_memory_projection(x: Any) -> Any:
    import mlx.core as mx

    x = mx.tanh(x)
    norm = mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True))
    return x / mx.maximum(norm, mx.array(1e-6, dtype=x.dtype))
