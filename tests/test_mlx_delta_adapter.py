import json
import sys
from types import ModuleType

import pytest


mx = pytest.importorskip("mlx.core")

from delta_mem_sidecar.mlx_delta_adapter import (
    attention_config_from_raw,
    convert_torch_adapter_to_mlx_npz,
    load_mlx_delta_adapter,
)


def test_attention_config_from_raw_maps_upstream_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DELTA_MEM_READ_SCALE", "1.0")
    config = attention_config_from_raw(
        {
            "rank": 8,
            "alpha": 16,
            "delta_heads": ["q", "o"],
            "normalize_qk": True,
            "couple_lambda": False,
            "state_update_mode": "lambda_outside",
        }
    )

    assert config.rank == 8
    assert config.alpha == 16
    assert config.active_delta_heads == frozenset({"q", "o"})
    assert config.normalize_qk is True
    assert config.couple_lambda is False
    assert config.state_update_mode == "lambda_outside"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("num_state_heads", 2, "num_state_heads=1"),
        ("num_memory_partitions", 2, "num_memory_partitions=1"),
        ("memory_write_granularity", "message_mean", "token write granularity"),
        ("delta_o_rmsnorm", True, "delta_o_rmsnorm"),
        ("trainable_delta_scale", True, "trainable_delta_scale"),
        ("delta_heads", ["q", "k", "o"], "q/o delta heads"),
    ],
)
def test_attention_config_from_raw_rejects_unsupported_mlx_variants(
    field: str,
    value,
    message: str,
) -> None:
    raw = {"rank": 2, "alpha": 4, "delta_heads": ["q", "o"]}
    raw[field] = value

    with pytest.raises(ValueError, match=message):
        attention_config_from_raw(raw)


def test_load_mlx_delta_adapter_converts_torch_state_dict(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DELTA_MEM_READ_SCALE", "1.0")
    (tmp_path / "delta_mem_config.json").write_text(
        json.dumps({"rank": 2, "alpha": 4, "delta_heads": ["q", "o"]})
    )
    (tmp_path / "delta_mem_adapter.pt").write_bytes(b"placeholder")
    torch = ModuleType("torch")

    def load(path, map_location=None, weights_only=False):
        assert path == tmp_path / "delta_mem_adapter.pt"
        assert map_location == "cpu"
        assert weights_only is True
        return {
            "model.layers.3.self_attn.memory_q_proj": [[1.0, 2.0]],
            "model.layers.3.self_attn.memory_k_proj": [[3.0, 4.0]],
            "model.layers.3.self_attn.memory_v_proj": [[5.0, 6.0]],
            "model.layers.3.self_attn.delta_q_proj": [[7.0], [8.0]],
            "model.layers.3.self_attn.delta_o_proj": [[9.0], [10.0]],
            "model.layers.3.self_attn.beta_proj": [[0.0, 0.0]],
            "model.layers.3.self_attn.beta_bias": [0.0],
            "unrelated.weight": [[99.0]],
        }

    torch.load = load
    monkeypatch.setitem(sys.modules, "torch", torch)

    adapter = load_mlx_delta_adapter(tmp_path)

    assert adapter.config.rank == 2
    assert adapter.config.alpha == 4
    assert sorted(adapter.weights_by_layer) == [3]
    weights = adapter.weights_by_layer[3]
    assert weights.memory_q_proj.tolist() == [[1.0, 2.0]]
    assert weights.delta_o_proj.tolist() == [[9.0], [10.0]]


def test_load_mlx_delta_adapter_prefers_mlx_npz_without_torch(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "delta_mem_config.json").write_text(
        json.dumps({"rank": 2, "alpha": 4, "delta_heads": ["q", "o"]})
    )
    mx.savez(
        str(tmp_path / "delta_mem_adapter_mlx.npz"),
        **{
            "model.layers.1.self_attn.memory_q_proj": mx.array([[1.0, 2.0]]),
            "model.layers.1.self_attn.memory_k_proj": mx.array([[3.0, 4.0]]),
            "model.layers.1.self_attn.memory_v_proj": mx.array([[5.0, 6.0]]),
            "model.layers.1.self_attn.delta_q_proj": mx.array([[7.0], [8.0]]),
            "model.layers.1.self_attn.delta_o_proj": mx.array([[9.0], [10.0]]),
            "model.layers.1.self_attn.beta_proj": mx.array([[0.0, 0.0]]),
            "model.layers.1.self_attn.beta_bias": mx.array([0.0]),
        },
    )
    monkeypatch.setitem(sys.modules, "torch", None)

    adapter = load_mlx_delta_adapter(tmp_path)

    assert sorted(adapter.weights_by_layer) == [1]
    assert adapter.weights_by_layer[1].memory_k_proj.tolist() == [[3.0, 4.0]]


def test_load_mlx_delta_adapter_accepts_output_only_linear_weights(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "delta_mem_config.json").write_text(
        json.dumps({"rank": 2, "alpha": 4, "delta_heads": ["q", "o"]})
    )
    mx.savez(
        str(tmp_path / "delta_mem_adapter_mlx.npz"),
        **{
            "model.layers.0.linear_attn.memory_q_proj": mx.zeros((2, 8)),
            "model.layers.0.linear_attn.memory_k_proj": mx.zeros((2, 8)),
            "model.layers.0.linear_attn.memory_v_proj": mx.zeros((2, 8)),
            "model.layers.0.linear_attn.delta_o_proj": mx.zeros((8, 2)),
            "model.layers.0.linear_attn.beta_proj": mx.zeros((2, 8)),
            "model.layers.0.linear_attn.beta_bias": mx.zeros((2,)),
        },
    )
    monkeypatch.setitem(sys.modules, "torch", None)

    adapter = load_mlx_delta_adapter(tmp_path)

    assert adapter.weights_by_layer == {}
    assert sorted(adapter.output_weights_by_layer) == [0]
    assert adapter.output_weights_by_layer[0].delta_o_proj.shape == (8, 2)


def test_load_mlx_delta_adapter_accepts_qwen35_full_attention_rank2_layers(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_attention_layers = [3, 7, 11, 15, 19, 23, 27, 31]
    hidden_size = 4096
    rank = 2
    (tmp_path / "delta_mem_config.json").write_text(
        json.dumps({"rank": rank, "alpha": 4, "delta_heads": ["q", "o"]})
    )
    arrays = {}
    for layer in full_attention_layers:
        prefix = f"model.layers.{layer}.self_attn"
        arrays.update(
            {
                f"{prefix}.memory_q_proj": mx.zeros((rank, hidden_size)),
                f"{prefix}.memory_k_proj": mx.zeros((rank, hidden_size)),
                f"{prefix}.memory_v_proj": mx.zeros((rank, hidden_size)),
                f"{prefix}.delta_q_proj": mx.zeros((hidden_size, rank)),
                f"{prefix}.delta_o_proj": mx.zeros((hidden_size, rank)),
                f"{prefix}.beta_proj": mx.zeros((rank, hidden_size)),
                f"{prefix}.beta_bias": mx.zeros((rank,)),
            }
        )
    mx.savez(str(tmp_path / "delta_mem_adapter_mlx.npz"), **arrays)
    monkeypatch.setitem(sys.modules, "torch", None)

    adapter = load_mlx_delta_adapter(tmp_path)

    assert adapter.config.rank == rank
    assert sorted(adapter.weights_by_layer) == full_attention_layers
    for layer in full_attention_layers:
        weights = adapter.weights_by_layer[layer]
        assert weights.memory_q_proj.shape == (rank, hidden_size)
        assert weights.memory_k_proj.shape == (rank, hidden_size)
        assert weights.memory_v_proj.shape == (rank, hidden_size)
        assert weights.delta_q_proj.shape == (hidden_size, rank)
        assert weights.delta_o_proj.shape == (hidden_size, rank)
        assert weights.beta_proj.shape == (rank, hidden_size)
        assert weights.beta_bias.shape == (rank,)


def test_convert_torch_adapter_to_mlx_npz_writes_runtime_artifact(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "delta_mem_adapter.pt").write_bytes(b"placeholder")
    torch = ModuleType("torch")
    torch.load = lambda *args, **kwargs: {
        "model.layers.2.self_attn.memory_q_proj": [[1.0, 2.0]],
        "model.layers.2.self_attn.memory_k_proj": [[3.0, 4.0]],
        "unrelated.weight": [[99.0]],
    }
    monkeypatch.setitem(sys.modules, "torch", torch)

    output_path = convert_torch_adapter_to_mlx_npz(tmp_path)

    loaded = dict(mx.load(str(output_path)))
    assert sorted(loaded) == [
        "model.layers.2.self_attn.memory_k_proj",
        "model.layers.2.self_attn.memory_q_proj",
    ]


def test_load_mlx_delta_adapter_requires_complete_layer_weights(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "delta_mem_config.json").write_text(json.dumps({"rank": 2}))
    (tmp_path / "delta_mem_adapter.pt").write_bytes(b"placeholder")
    torch = ModuleType("torch")
    torch.load = lambda *args, **kwargs: {
        "model.layers.0.self_attn.memory_q_proj": [[1.0, 2.0]],
    }
    monkeypatch.setitem(sys.modules, "torch", torch)

    with pytest.raises(ValueError, match="missing"):
        load_mlx_delta_adapter(tmp_path)
