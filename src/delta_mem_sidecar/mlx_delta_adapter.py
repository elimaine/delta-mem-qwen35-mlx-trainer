from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from delta_mem_sidecar.mlx_delta_attention import (
    MlxDeltaAttentionConfig,
    MlxDeltaAttentionWeights,
    MlxDeltaOutputWeights,
)


_ADAPTER_KEY_RE = re.compile(
    r"(?:^|\.)layers\.(?P<layer>\d+)\.self_attn\."
    r"(?P<name>memory_q_proj|memory_k_proj|memory_v_proj|delta_q_proj|"
    r"delta_o_proj|beta_proj|beta_bias|lambda_proj|lambda_bias)$"
)
_OUTPUT_KEY_RE = re.compile(
    r"(?:^|\.)layers\.(?P<layer>\d+)\.linear_attn\."
    r"(?P<name>memory_q_proj|memory_k_proj|memory_v_proj|"
    r"delta_o_proj|beta_proj|beta_bias|lambda_proj|lambda_bias)$"
)


@dataclass(frozen=True)
class MlxDeltaAdapter:
    config: MlxDeltaAttentionConfig
    weights_by_layer: dict[int, MlxDeltaAttentionWeights]
    output_weights_by_layer: dict[int, MlxDeltaOutputWeights]
    raw_config: dict[str, Any]


def load_mlx_delta_adapter(adapter_dir: str | Path) -> MlxDeltaAdapter:
    adapter_path = Path(adapter_dir)
    raw_config = json.loads((adapter_path / "delta_mem_config.json").read_text())
    config = attention_config_from_raw(raw_config)
    state_dict = _load_adapter_state_dict(adapter_path)
    grouped: dict[int, dict[str, Any]] = {}
    output_grouped: dict[int, dict[str, Any]] = {}
    for key, tensor in state_dict.items():
        match = _ADAPTER_KEY_RE.search(key)
        if match is not None:
            grouped.setdefault(int(match.group("layer")), {})[match.group("name")] = _to_mlx_array(tensor)
            continue
        output_match = _OUTPUT_KEY_RE.search(key)
        if output_match is not None:
            output_grouped.setdefault(int(output_match.group("layer")), {})[output_match.group("name")] = _to_mlx_array(tensor)

    weights_by_layer = {
        layer: _weights_from_group(layer, group)
        for layer, group in sorted(grouped.items())
    }
    output_weights_by_layer = {
        layer: _output_weights_from_group(layer, group)
        for layer, group in sorted(output_grouped.items())
    }
    if not weights_by_layer and not output_weights_by_layer:
        raise ValueError(f"no supported δ-mem weights found in {adapter_path}")
    return MlxDeltaAdapter(
        config=config,
        weights_by_layer=weights_by_layer,
        output_weights_by_layer=output_weights_by_layer,
        raw_config=raw_config,
    )


def convert_torch_adapter_to_mlx_npz(
    adapter_dir: str | Path,
    *,
    output_path: str | Path | None = None,
) -> Path:
    adapter_path = Path(adapter_dir)
    target_path = Path(output_path) if output_path is not None else adapter_path / "delta_mem_adapter_mlx.npz"
    state_dict = _load_torch_state_dict(adapter_path / "delta_mem_adapter.pt")
    arrays = {
        key: _to_mlx_array(tensor)
        for key, tensor in state_dict.items()
        if _ADAPTER_KEY_RE.search(key) is not None or _OUTPUT_KEY_RE.search(key) is not None
    }
    if not arrays:
        raise ValueError(f"no supported δ-mem weights found in {adapter_path}")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    _save_mlx_npz(target_path, arrays)
    return target_path


def attention_config_from_raw(raw_config: dict[str, Any]) -> MlxDeltaAttentionConfig:
    _validate_mlx_supported_raw_config(raw_config)
    read_scale = float(os.getenv("DELTA_MEM_READ_SCALE", "0.5"))
    max_delta_ratio_raw = os.getenv("DELTA_MEM_MAX_DELTA_RATIO", "0.5")
    max_delta_ratio = None if max_delta_ratio_raw.strip().lower() in {"", "off", "none", "0"} else float(max_delta_ratio_raw)
    return MlxDeltaAttentionConfig(
        rank=int(raw_config["rank"]),
        alpha=float(raw_config.get("alpha", raw_config["rank"])) * read_scale,
        q_delta_scale=float(os.getenv("DELTA_MEM_Q_DELTA_SCALE", "1.0")),
        o_delta_scale=float(os.getenv("DELTA_MEM_O_DELTA_SCALE", "1.0")),
        max_delta_ratio=max_delta_ratio,
        active_delta_heads=frozenset(raw_config.get("delta_heads", ("q", "o"))),
        normalize_qk=bool(raw_config.get("normalize_qk", True)),
        couple_lambda=bool(raw_config.get("couple_lambda", True)),
        state_update_mode=str(raw_config.get("state_update_mode", "standard")),
    )


def _validate_mlx_supported_raw_config(raw_config: dict[str, Any]) -> None:
    """Reject upstream delta-Mem variants this MLX runtime cannot execute yet."""

    if int(raw_config.get("num_state_heads", 1)) != 1:
        raise ValueError("MLX δ-mem runtime currently supports num_state_heads=1 only")
    if int(raw_config.get("num_memory_partitions", 1)) != 1:
        raise ValueError("MLX δ-mem runtime currently supports num_memory_partitions=1 only")
    if int(raw_config.get("num_global_memory_partitions", 0)) != 0:
        raise ValueError("MLX δ-mem runtime does not support global memory partitions")
    if str(raw_config.get("memory_readout_mode", "delta")) != "delta":
        raise ValueError("MLX δ-mem runtime currently supports memory_readout_mode='delta' only")
    if str(raw_config.get("memory_write_source", "learned_hidden")) != "learned_hidden":
        raise ValueError("MLX δ-mem runtime currently supports memory_write_source='learned_hidden' only")
    if str(raw_config.get("memory_write_granularity", "token")) != "token":
        raise ValueError("MLX δ-mem runtime currently supports token write granularity only")
    if bool(raw_config.get("delta_o_rmsnorm", False)):
        raise ValueError("MLX δ-mem runtime does not support delta_o_rmsnorm adapters yet")
    if bool(raw_config.get("trainable_delta_scale", False)):
        raise ValueError("MLX δ-mem runtime does not support trainable_delta_scale adapters yet")

    delta_heads = frozenset(raw_config.get("delta_heads", ("q", "o")))
    unsupported_heads = delta_heads - {"q", "o"}
    if unsupported_heads:
        raise ValueError(
            "MLX δ-mem runtime currently supports only q/o delta heads; "
            f"unsupported: {', '.join(sorted(unsupported_heads))}"
        )


def _load_adapter_state_dict(adapter_path: Path) -> dict[str, Any]:
    mlx_path = adapter_path / "delta_mem_adapter_mlx.npz"
    if mlx_path.exists():
        return _load_mlx_npz(mlx_path)
    return _load_torch_state_dict(adapter_path / "delta_mem_adapter.pt")


def _load_torch_state_dict(path: Path) -> dict[str, Any]:
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "Loading `delta_mem_adapter.pt` currently requires PyTorch for the "
            "one-time Torch checkpoint read. Install torch in the conversion "
            "environment, then convert the tensors to MLX arrays."
        ) from exc
    return torch.load(path, map_location="cpu", weights_only=True)


def _load_mlx_npz(path: Path) -> dict[str, Any]:
    import mlx.core as mx

    return dict(mx.load(str(path)))


def _save_mlx_npz(path: Path, arrays: dict[str, Any]) -> None:
    import mlx.core as mx

    mx.savez(str(path), **arrays)


def _weights_from_group(layer: int, group: dict[str, Any]) -> MlxDeltaAttentionWeights:
    required = {
        "memory_q_proj",
        "memory_k_proj",
        "memory_v_proj",
        "delta_q_proj",
        "delta_o_proj",
        "beta_proj",
        "beta_bias",
    }
    missing = sorted(required - set(group))
    if missing:
        raise ValueError(f"layer {layer} is missing δ-mem weights: {', '.join(missing)}")
    return MlxDeltaAttentionWeights(
        memory_q_proj=group["memory_q_proj"],
        memory_k_proj=group["memory_k_proj"],
        memory_v_proj=group["memory_v_proj"],
        delta_q_proj=group["delta_q_proj"],
        delta_o_proj=group["delta_o_proj"],
        beta_proj=group["beta_proj"],
        beta_bias=group["beta_bias"],
        lambda_proj=group.get("lambda_proj"),
        lambda_bias=group.get("lambda_bias"),
    )


def _output_weights_from_group(layer: int, group: dict[str, Any]) -> MlxDeltaOutputWeights:
    required = {
        "memory_q_proj",
        "memory_k_proj",
        "memory_v_proj",
        "delta_o_proj",
        "beta_proj",
        "beta_bias",
    }
    missing = sorted(required - set(group))
    if missing:
        raise ValueError(f"linear layer {layer} is missing δ-mem output weights: {', '.join(missing)}")
    return MlxDeltaOutputWeights(
        memory_q_proj=group["memory_q_proj"],
        memory_k_proj=group["memory_k_proj"],
        memory_v_proj=group["memory_v_proj"],
        delta_o_proj=group["delta_o_proj"],
        beta_proj=group["beta_proj"],
        beta_bias=group["beta_bias"],
        lambda_proj=group.get("lambda_proj"),
        lambda_bias=group.get("lambda_bias"),
    )


def _to_mlx_array(tensor: Any) -> Any:
    import mlx.core as mx

    if hasattr(tensor, "detach"):
        tensor = tensor.detach()
    if hasattr(tensor, "float"):
        tensor = tensor.float()
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    if hasattr(tensor, "numpy"):
        tensor = tensor.numpy()
    return mx.array(tensor)
