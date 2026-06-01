import math

import pytest


mx = pytest.importorskip("mlx.core")

from mlx_lm.models.qwen2 import Attention, ModelArgs

from delta_mem_sidecar.mlx_delta_attention import (
    MlxDeltaAttention,
    MlxDeltaAttentionConfig,
    MlxDeltaAttentionWeights,
    MlxDeltaGatedDeltaNetOutput,
    MlxDeltaOutputWeights,
    get_mlx_delta_state,
    infer_mlx_attention_shape,
    iter_mlx_delta_output_modules,
    reset_mlx_delta_states,
    wrap_qwen_gated_delta_output_layers,
    wrap_qwen_attention_layers,
)


def test_zero_delta_attention_matches_base_attention() -> None:
    base = _tiny_attention()
    weights = _weights(hidden_size=8, rank=2)
    config = MlxDeltaAttentionConfig(rank=2, active_delta_heads=frozenset({"q", "o"}))
    wrapped = MlxDeltaAttention(base, weights, config)
    x = mx.ones((1, 3, 8))

    base_output = base(x)
    wrapped_output = wrapped(x)
    mx.eval(base_output, wrapped_output)

    assert mx.max(mx.abs(base_output - wrapped_output)).item() < 1e-6
    assert wrapped.delta_state is not None


def test_delta_attention_can_change_output_and_reset_state() -> None:
    base = _tiny_attention()
    weights = _weights(
        hidden_size=8,
        rank=2,
        memory_fill=0.01,
        delta_q_fill=0.01,
        delta_o_fill=0.02,
    )
    config = MlxDeltaAttentionConfig(rank=2, active_delta_heads=frozenset({"q", "o"}))
    wrapped = MlxDeltaAttention(base, weights, config)
    x = mx.ones((1, 3, 8))

    base_output = base(x)
    wrapped_output = wrapped(x)
    mx.eval(base_output, wrapped_output)

    assert mx.max(mx.abs(base_output - wrapped_output)).item() > 0
    wrapped.reset_state()
    assert wrapped.delta_state is None


def test_wrap_qwen_attention_layers_replaces_selected_layers() -> None:
    model = type("Model", (), {})()
    model.model = type("Inner", (), {})()
    model.model.layers = [type("Layer", (), {})(), type("Layer", (), {})()]
    model.model.layers[0].self_attn = _tiny_attention()
    model.model.layers[1].self_attn = _tiny_attention()
    weights = _weights(hidden_size=8, rank=2)

    wrapped = wrap_qwen_attention_layers(
        model,
        weights_by_layer={1: weights},
        config=MlxDeltaAttentionConfig(rank=2),
    )

    assert wrapped == [1]
    assert isinstance(model.model.layers[1].self_attn, MlxDeltaAttention)
    assert not isinstance(model.model.layers[0].self_attn, MlxDeltaAttention)
    assert get_mlx_delta_state(model) == {}
    reset_mlx_delta_states(model)


def test_memory_update_coefficients_follow_upstream_modes() -> None:
    wrapped = MlxDeltaAttention(
        _tiny_attention(),
        _weights(hidden_size=8, rank=2),
        MlxDeltaAttentionConfig(rank=2, state_update_mode="standard"),
    )
    beta = mx.array([[[0.25, 0.5]]])
    lam = mx.array([[[0.75, 0.5]]])

    keep, erase, write = wrapped._memory_update_coefficients(beta, lam)

    assert keep.tolist() == lam.tolist()
    assert erase.tolist() == beta.tolist()
    assert write.tolist() == beta.tolist()

    wrapped = MlxDeltaAttention(
        _tiny_attention(),
        _weights(hidden_size=8, rank=2),
        MlxDeltaAttentionConfig(rank=2, state_update_mode="lambda_outside"),
    )

    keep, erase, write = wrapped._memory_update_coefficients(beta, lam)

    assert keep.tolist() == lam.tolist()
    assert erase.tolist() == [[[0.1875, 0.25]]]
    assert write.tolist() == beta.tolist()


def test_qwen35_shape_detection_accepts_effective_query_width() -> None:
    base = _fake_qwen35_attention()
    weights = _weights(hidden_size=8, rank=2)

    shape = infer_mlx_attention_shape(base, weights)
    wrapped = MlxDeltaAttention(base, weights, MlxDeltaAttentionConfig(rank=2))

    assert shape.is_query_gated is True
    assert shape.q_proj_out_dims == 16
    assert shape.query_out_dims == 8
    assert shape.n_heads == 2
    assert shape.n_kv_heads == 1
    assert shape.head_dim == 4
    assert wrapped.is_query_gated is True


def test_qwen35_query_gate_is_split_and_applied_before_o_proj(monkeypatch: pytest.MonkeyPatch) -> None:
    from delta_mem_sidecar import mlx_delta_attention

    def fake_attention(queries, keys, values, *, cache=None, scale=None, mask=None):
        return mx.ones_like(queries)

    monkeypatch.setattr(mlx_delta_attention, "_scaled_dot_product_attention", fake_attention)
    base = _fake_qwen35_attention(query_fill=0.0, gate_fill=2.0)
    weights = _weights(hidden_size=8, rank=2)
    wrapped = MlxDeltaAttention(base, weights, MlxDeltaAttentionConfig(rank=2))

    output = wrapped(mx.ones((1, 2, 8)))
    mx.eval(output)

    expected = mx.full((1, 2, 8), 1.0 / (1.0 + math.exp(-2.0)))
    assert mx.max(mx.abs(output - expected)).item() < 1e-6
    assert wrapped.last_query_gate is not None
    assert wrapped.last_query_gate.tolist() == mx.full((1, 2, 8), 2.0).tolist()


def test_qwen35_delta_q_shape_must_match_query_half() -> None:
    base = _fake_qwen35_attention()
    weights = _weights(hidden_size=8, rank=2)
    bad_weights = MlxDeltaAttentionWeights(
        memory_q_proj=weights.memory_q_proj,
        memory_k_proj=weights.memory_k_proj,
        memory_v_proj=weights.memory_v_proj,
        delta_q_proj=mx.zeros((16, 2)),
        delta_o_proj=weights.delta_o_proj,
        beta_proj=weights.beta_proj,
        beta_bias=weights.beta_bias,
    )

    with pytest.raises(RuntimeError, match="q output"):
        MlxDeltaAttention(base, bad_weights, MlxDeltaAttentionConfig(rank=2))


def test_gated_delta_output_zero_delta_matches_base_output() -> None:
    base = _fake_gated_delta_net()
    wrapped = MlxDeltaGatedDeltaNetOutput(
        base,
        _output_weights(hidden_size=8, rank=2),
        MlxDeltaAttentionConfig(rank=2),
    )
    x = mx.ones((1, 3, 8))

    base_output = base(x)
    wrapped_output = wrapped(x)
    mx.eval(base_output, wrapped_output)

    assert mx.max(mx.abs(base_output - wrapped_output)).item() < 1e-6
    assert wrapped.delta_state is not None


def test_gated_delta_output_delta_can_change_output_and_wrap_layers() -> None:
    model = type("Model", (), {})()
    model.model = type("Inner", (), {})()
    model.model.layers = [type("Layer", (), {})(), type("Layer", (), {})()]
    model.model.layers[0].is_linear = True
    model.model.layers[0].linear_attn = _fake_gated_delta_net()
    model.model.layers[1].is_linear = False

    wrapped_layers = wrap_qwen_gated_delta_output_layers(
        model,
        weights_by_layer={
            0: _output_weights(hidden_size=8, rank=2, memory_fill=0.01, delta_o_fill=0.02),
        },
        config=MlxDeltaAttentionConfig(rank=2),
    )
    wrapped = dict(iter_mlx_delta_output_modules(model))
    output = model.model.layers[0].linear_attn(mx.ones((1, 3, 8)))
    mx.eval(output)

    assert wrapped_layers == [0]
    assert list(wrapped) == [0]
    assert mx.max(mx.abs(output - mx.full((1, 3, 8), 2.0))).item() > 0
    reset_mlx_delta_states(model)


def _tiny_attention() -> Attention:
    return Attention(
        ModelArgs(
            model_type="qwen2",
            hidden_size=8,
            num_hidden_layers=1,
            intermediate_size=16,
            num_attention_heads=2,
            rms_norm_eps=1e-6,
            vocab_size=16,
            num_key_value_heads=1,
            max_position_embeddings=16,
        )
    )


class _FakeLinear:
    def __init__(self, output_dims: int, fill: float = 0.0) -> None:
        self.output_dims = output_dims
        self.fill = fill

    def __call__(self, x):
        return mx.full((*x.shape[:-1], self.output_dims), self.fill, dtype=x.dtype)


class _FakeQwen35QProj:
    output_dims = 16

    def __init__(self, *, query_fill: float, gate_fill: float) -> None:
        self.query_fill = query_fill
        self.gate_fill = gate_fill

    def __call__(self, x):
        query = mx.full((*x.shape[:-1], 2, 4), self.query_fill, dtype=x.dtype)
        gate = mx.full((*x.shape[:-1], 2, 4), self.gate_fill, dtype=x.dtype)
        return mx.concatenate([query, gate], axis=-1).reshape(*x.shape[:-1], 16)


class _IdentityRope:
    def __call__(self, x, offset=None):
        return x


class _IdentityLinear:
    output_dims = 8

    def __call__(self, x):
        return x


class _FakeGatedDeltaNet:
    hidden_size = 8

    def __call__(self, inputs, mask=None, cache=None):
        return mx.full(inputs.shape, 2.0, dtype=inputs.dtype)


def _fake_gated_delta_net():
    return _FakeGatedDeltaNet()


def _fake_qwen35_attention(*, query_fill: float = 0.0, gate_fill: float = 0.0):
    attention = type("FakeQwen35Attention", (), {})()
    attention.num_key_value_heads = 1
    attention.head_dim = 4
    attention.scale = 1.0
    attention.q_proj = _FakeQwen35QProj(query_fill=query_fill, gate_fill=gate_fill)
    attention.k_proj = _FakeLinear(4)
    attention.v_proj = _FakeLinear(4)
    attention.o_proj = _IdentityLinear()
    attention.rope = _IdentityRope()
    attention.q_norm = None
    attention.k_norm = None
    return attention


def _weights(
    *,
    hidden_size: int,
    rank: int,
    memory_fill: float = 0.0,
    delta_q_fill: float = 0.0,
    delta_o_fill: float = 0.0,
) -> MlxDeltaAttentionWeights:
    return MlxDeltaAttentionWeights(
        memory_q_proj=mx.full((rank, hidden_size), memory_fill),
        memory_k_proj=mx.full((rank, hidden_size), memory_fill),
        memory_v_proj=mx.full((rank, hidden_size), memory_fill),
        delta_q_proj=mx.full((hidden_size, rank), delta_q_fill),
        delta_o_proj=mx.full((hidden_size, rank), delta_o_fill),
        beta_proj=mx.zeros((rank, hidden_size)),
        beta_bias=mx.zeros((rank,)),
    )


def _output_weights(
    *,
    hidden_size: int,
    rank: int,
    memory_fill: float = 0.0,
    delta_o_fill: float = 0.0,
) -> MlxDeltaOutputWeights:
    return MlxDeltaOutputWeights(
        memory_q_proj=mx.full((rank, hidden_size), memory_fill),
        memory_k_proj=mx.full((rank, hidden_size), memory_fill),
        memory_v_proj=mx.full((rank, hidden_size), memory_fill),
        delta_o_proj=mx.full((hidden_size, rank), delta_o_fill),
        beta_proj=mx.zeros((rank, hidden_size)),
        beta_bias=mx.zeros((rank,)),
    )
