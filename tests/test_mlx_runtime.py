import sys
from types import ModuleType

import pytest

from delta_mem_sidecar.mlx_delta_attention import MlxDeltaAttentionWeights, MlxDeltaOutputWeights
from delta_mem_sidecar.mlx_runtime import (
    MlxBackboneRuntime,
    _validate_adapter_compatible,
    _validate_output_adapter_compatible,
)
from delta_mem_sidecar.runtime import ChatMessage


def test_mlx_runtime_missing_dependency_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "mlx_lm", None)
    runtime = MlxBackboneRuntime(model_path="mlx-community/test-model")

    with pytest.raises(RuntimeError, match="MLX runtime is not installed"):
        runtime.generate(
            messages=[ChatMessage(role="user", content="hello")],
            state=runtime.fresh_state(),
        )


def test_mlx_runtime_generates_with_lazy_loaded_mlx_lm(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("mlx_lm")
    sample_utils = ModuleType("mlx_lm.sample_utils")
    calls: dict[str, object] = {}

    class Tokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            calls["messages"] = messages
            calls["tokenize"] = tokenize
            calls["add_generation_prompt"] = add_generation_prompt
            return "rendered prompt"

    def load(model_path: str):
        calls["model_path"] = model_path
        return "model", Tokenizer()

    def make_sampler(temp: float):
        return f"sampler:{temp}"

    def generate(model, tokenizer, *, prompt: str, verbose: bool, max_tokens: int, sampler):
        calls["generate"] = {
            "model": model,
            "prompt": prompt,
            "verbose": verbose,
            "max_tokens": max_tokens,
            "sampler": sampler,
        }
        return "mlx response"

    module.load = load
    module.generate = generate
    sample_utils.make_sampler = make_sampler
    monkeypatch.setitem(sys.modules, "mlx_lm", module)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_utils)

    runtime = MlxBackboneRuntime(model_path="mlx-community/test-model")
    state = runtime.fresh_state()
    result = runtime.generate(
        messages=[ChatMessage(role="user", content="hello")],
        state=state,
        max_tokens=7,
        temperature=0.2,
    )

    assert result.content == "mlx response"
    assert state.updates == 1
    assert calls["model_path"] == "mlx-community/test-model"
    assert calls["messages"] == [{"role": "user", "content": "hello"}]
    assert calls["add_generation_prompt"] is True
    assert calls["generate"] == {
        "model": "model",
        "prompt": "rendered prompt",
        "verbose": False,
        "max_tokens": 7,
        "sampler": "sampler:0.2",
    }


def test_mlx_runtime_passes_generation_kwargs_when_generate_accepts_var_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("mlx_lm")
    sample_utils = ModuleType("mlx_lm.sample_utils")
    calls: dict[str, object] = {}

    class Tokenizer:
        pass

    def load(model_path: str):
        return "model", Tokenizer()

    def generate(model, tokenizer, *, prompt: str, verbose: bool, **kwargs):
        calls["kwargs"] = kwargs
        return "ok"

    def make_sampler(temp: float):
        return f"sampler:{temp}"

    module.load = load
    module.generate = generate
    sample_utils.make_sampler = make_sampler
    monkeypatch.setitem(sys.modules, "mlx_lm", module)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_utils)

    runtime = MlxBackboneRuntime(model_path="mlx-community/test-model")
    runtime.generate(
        messages=[ChatMessage(role="user", content="hello")],
        state=runtime.fresh_state(),
        max_tokens=3,
        temperature=0.0,
    )

    assert calls["kwargs"] == {"max_tokens": 3}


def test_validate_adapter_compatible_rejects_hidden_size_mismatch() -> None:
    model = type("Model", (), {})()
    model.model = type("Inner", (), {})()
    layer = type("Layer", (), {})()
    attention = type("FakeQwen35Attention", (), {})()
    attention.q_proj = type("Linear", (), {"input_dims": 8, "output_dims": 8})()
    attention.o_proj = type("Linear", (), {"output_dims": 8})()
    layer.self_attn = attention
    model.model.layers = [layer]
    weights = MlxDeltaAttentionWeights(
        memory_q_proj=type("Shape", (), {"shape": (2, 9)})(),
        memory_k_proj=type("Shape", (), {"shape": (2, 9)})(),
        memory_v_proj=type("Shape", (), {"shape": (2, 9)})(),
        delta_q_proj=type("Shape", (), {"shape": (8, 2)})(),
        delta_o_proj=type("Shape", (), {"shape": (8, 2)})(),
        beta_proj=type("Shape", (), {"shape": (2, 9)})(),
        beta_bias=type("Shape", (), {"shape": (2,)})(),
    )

    with pytest.raises(RuntimeError, match="hidden size mismatch"):
        _validate_adapter_compatible(model, {0: weights})


def test_validate_adapter_compatible_accepts_qwen35_doubled_q_projection() -> None:
    model = type("Model", (), {})()
    model.model = type("Inner", (), {})()
    layer = type("Layer", (), {})()
    attention = type("FakeQwen35Attention", (), {})()
    attention.q_proj = type("Linear", (), {"output_dims": 16})()
    attention.o_proj = type("Linear", (), {"output_dims": 8})()
    attention.num_key_value_heads = 1
    attention.head_dim = 4
    layer.self_attn = attention
    model.model.layers = [layer]
    weights = _weights(hidden_size=8, rank=2)

    _validate_adapter_compatible(model, {0: weights})


def test_validate_adapter_compatible_rejects_qwen35_doubled_delta_q() -> None:
    model = type("Model", (), {})()
    model.model = type("Inner", (), {})()
    layer = type("Layer", (), {})()
    attention = type("FakeQwen35Attention", (), {})()
    attention.q_proj = type("Linear", (), {"output_dims": 16})()
    attention.o_proj = type("Linear", (), {"output_dims": 8})()
    attention.num_key_value_heads = 1
    attention.head_dim = 4
    layer.self_attn = attention
    model.model.layers = [layer]
    weights = MlxDeltaAttentionWeights(
        memory_q_proj=type("Shape", (), {"shape": (2, 8)})(),
        memory_k_proj=type("Shape", (), {"shape": (2, 8)})(),
        memory_v_proj=type("Shape", (), {"shape": (2, 8)})(),
        delta_q_proj=type("Shape", (), {"shape": (16, 2)})(),
        delta_o_proj=type("Shape", (), {"shape": (8, 2)})(),
        beta_proj=type("Shape", (), {"shape": (2, 8)})(),
        beta_bias=type("Shape", (), {"shape": (2,)})(),
    )

    with pytest.raises(RuntimeError, match="q output"):
        _validate_adapter_compatible(model, {0: weights})


def test_validate_output_adapter_compatible_accepts_linear_layer() -> None:
    model = type("Model", (), {})()
    model.model = type("Inner", (), {})()
    layer = type("Layer", (), {})()
    layer.is_linear = True
    layer.linear_attn = type("FakeGatedDeltaNet", (), {"hidden_size": 8})()
    model.model.layers = [layer]

    _validate_output_adapter_compatible(model, {0: _output_weights(hidden_size=8, rank=2)})


def test_validate_output_adapter_compatible_rejects_full_attention_layer() -> None:
    model = type("Model", (), {})()
    model.model = type("Inner", (), {})()
    layer = type("Layer", (), {})()
    layer.is_linear = False
    layer.self_attn = object()
    model.model.layers = [layer]

    with pytest.raises(RuntimeError, match="not a GatedDeltaNet"):
        _validate_output_adapter_compatible(model, {0: _output_weights(hidden_size=8, rank=2)})


def _weights(*, hidden_size: int, rank: int) -> MlxDeltaAttentionWeights:
    return MlxDeltaAttentionWeights(
        memory_q_proj=type("Shape", (), {"shape": (rank, hidden_size)})(),
        memory_k_proj=type("Shape", (), {"shape": (rank, hidden_size)})(),
        memory_v_proj=type("Shape", (), {"shape": (rank, hidden_size)})(),
        delta_q_proj=type("Shape", (), {"shape": (hidden_size, rank)})(),
        delta_o_proj=type("Shape", (), {"shape": (hidden_size, rank)})(),
        beta_proj=type("Shape", (), {"shape": (rank, hidden_size)})(),
        beta_bias=type("Shape", (), {"shape": (rank,)})(),
    )


def _output_weights(*, hidden_size: int, rank: int) -> MlxDeltaOutputWeights:
    return MlxDeltaOutputWeights(
        memory_q_proj=type("Shape", (), {"shape": (rank, hidden_size)})(),
        memory_k_proj=type("Shape", (), {"shape": (rank, hidden_size)})(),
        memory_v_proj=type("Shape", (), {"shape": (rank, hidden_size)})(),
        delta_o_proj=type("Shape", (), {"shape": (hidden_size, rank)})(),
        beta_proj=type("Shape", (), {"shape": (rank, hidden_size)})(),
        beta_bias=type("Shape", (), {"shape": (rank,)})(),
    )
