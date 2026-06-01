#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path
from typing import Any

from delta_mem_sidecar.mlx_delta_attention import (
    MlxDeltaAttentionConfig,
    MlxDeltaAttentionWeights,
    MlxDeltaOutputWeights,
    infer_mlx_attention_shape,
    wrap_qwen_gated_delta_output_layers,
    wrap_qwen_attention_layers,
)


DEFAULT_MODEL = "mlx-community/Qwen3.5-9B-MLX-4bit"
FULL_ATTENTION_LAYERS = (3, 7, 11, 15, 19, 23, 27, 31)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check that zeroed δ-mem wrapping preserves real Qwen3.5 MLX logits."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--layers", default="all", help="Comma-separated layer ids or 'all'.")
    parser.add_argument(
        "--linear-layers",
        default="none",
        help="Comma-separated GatedDeltaNet layer ids, 'all', or 'none'.",
    )
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--prompt", default="Remember: Delta is local memory. What is Delta?")
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    if not args.allow_cpu:
        require_gpu(mx)

    layers = parse_full_attention_layers(args.layers)
    linear_layers = parse_linear_layers(args.linear_layers)
    report: dict[str, Any] = {
        "kind": "qwen35_zero_delta_parity",
        "model": args.model,
        "layers": layers,
        "linear_layers": linear_layers,
        "rank": args.rank,
        "alpha": args.alpha,
        "prompt": args.prompt,
        "atol": args.atol,
        "timing_seconds": {},
        "memory": {"initial": memory_snapshot(mx)},
    }

    start = time.perf_counter()
    model, tokenizer = load(args.model)
    report["timing_seconds"]["load_model"] = round(time.perf_counter() - start, 3)
    report["memory"]["after_model_load"] = memory_snapshot(mx)

    tokens = mx.array([tokenizer.encode(args.prompt)])
    start = time.perf_counter()
    base_logits = model(tokens)
    mx.eval(base_logits)
    report["timing_seconds"]["base_forward"] = round(time.perf_counter() - start, 3)

    weights = build_zero_weights(model, layers=layers, rank=args.rank, mx=mx)
    linear_weights = build_zero_linear_weights(model, layers=linear_layers, rank=args.rank, mx=mx)
    config = MlxDeltaAttentionConfig(
        rank=args.rank,
        alpha=args.alpha,
        active_delta_heads=frozenset({"q", "o"}),
        max_delta_ratio=None,
    )
    wrapped = wrap_qwen_attention_layers(model, weights_by_layer=weights, config=config)
    wrapped_linear = wrap_qwen_gated_delta_output_layers(
        model,
        weights_by_layer=linear_weights,
        config=config,
    )
    report["wrapped_layers"] = wrapped
    report["wrapped_linear_layers"] = wrapped_linear
    report["wrapped_details"] = wrapped_details(model, wrapped, wrapped_linear)

    start = time.perf_counter()
    wrapped_logits = model(tokens)
    mx.eval(wrapped_logits)
    report["timing_seconds"]["wrapped_forward"] = round(time.perf_counter() - start, 3)
    report["memory"]["after_wrapped_forward"] = memory_snapshot(mx)

    diff = mx.abs(base_logits.astype(mx.float32) - wrapped_logits.astype(mx.float32))
    max_abs = float(mx.max(diff).item())
    mean_abs = float(mx.mean(diff).item())
    report["logits_shape"] = list(base_logits.shape)
    report["max_abs_diff"] = max_abs
    report["mean_abs_diff"] = mean_abs
    report["passed"] = max_abs <= args.atol

    output = json.dumps(report, indent=2, sort_keys=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0 if report["passed"] else 2


def parse_full_attention_layers(raw: str) -> list[int]:
    if raw == "all":
        return list(FULL_ATTENTION_LAYERS)
    layers = [int(part) for part in raw.split(",") if part.strip()]
    for layer in layers:
        if layer not in FULL_ATTENTION_LAYERS:
            raise ValueError(f"Layer {layer} is not a known Qwen3.5 full-attention layer")
    return layers


def parse_linear_layers(raw: str) -> list[int]:
    if raw in {"", "none", "off"}:
        return []
    if raw == "all":
        return [layer for layer in range(32) if layer not in FULL_ATTENTION_LAYERS]
    layers = [int(part) for part in raw.split(",") if part.strip()]
    for layer in layers:
        if layer in FULL_ATTENTION_LAYERS:
            raise ValueError(f"Layer {layer} is a full-attention layer, not a GatedDeltaNet layer")
    return layers


def build_zero_weights(model: Any, *, layers: list[int], rank: int, mx: Any) -> dict[int, MlxDeltaAttentionWeights]:
    result = {}
    for layer_index in layers:
        attention = get_layers(model)[layer_index].self_attn
        shape = infer_mlx_attention_shape(
            attention,
            MlxDeltaAttentionWeights(
                memory_q_proj=mx.zeros((rank, infer_hidden_size(attention))),
                memory_k_proj=mx.zeros((rank, infer_hidden_size(attention))),
                memory_v_proj=mx.zeros((rank, infer_hidden_size(attention))),
                delta_q_proj=mx.zeros((infer_effective_query_width(attention), rank)),
                delta_o_proj=mx.zeros((infer_hidden_size(attention), rank)),
                beta_proj=mx.zeros((rank, infer_hidden_size(attention))),
                beta_bias=mx.zeros((rank,)),
            ),
        )
        hidden_size = infer_hidden_size(attention)
        result[layer_index] = MlxDeltaAttentionWeights(
            memory_q_proj=mx.zeros((rank, hidden_size)),
            memory_k_proj=mx.zeros((rank, hidden_size)),
            memory_v_proj=mx.zeros((rank, hidden_size)),
            delta_q_proj=mx.zeros((shape.query_out_dims, rank)),
            delta_o_proj=mx.zeros((hidden_size, rank)),
            beta_proj=mx.zeros((rank, hidden_size)),
            beta_bias=mx.zeros((rank,)),
        )
    return result


def build_zero_linear_weights(model: Any, *, layers: list[int], rank: int, mx: Any) -> dict[int, MlxDeltaOutputWeights]:
    result = {}
    for layer_index in layers:
        layer = get_layers(model)[layer_index]
        if not getattr(layer, "is_linear", False):
            raise RuntimeError(f"layer {layer_index} is not a linear GatedDeltaNet layer")
        attention = layer.linear_attn
        hidden_size = int(getattr(attention, "hidden_size"))
        result[layer_index] = MlxDeltaOutputWeights(
            memory_q_proj=mx.zeros((rank, hidden_size)),
            memory_k_proj=mx.zeros((rank, hidden_size)),
            memory_v_proj=mx.zeros((rank, hidden_size)),
            delta_o_proj=mx.zeros((hidden_size, rank)),
            beta_proj=mx.zeros((rank, hidden_size)),
            beta_bias=mx.zeros((rank,)),
        )
    return result


def wrapped_details(model: Any, attention_layers: list[int], linear_layers: list[int]) -> dict[str, list[dict[str, Any]]]:
    return {
        "attention": attention_wrapped_details(model, attention_layers),
        "linear_output": linear_wrapped_details(model, linear_layers),
    }


def attention_wrapped_details(model: Any, layers: list[int]) -> list[dict[str, Any]]:
    details = []
    for layer_index in layers:
        attention = get_layers(model)[layer_index].self_attn
        details.append(
            {
                "layer": layer_index,
                "attention_class": type(attention.base).__name__,
                "is_query_gated": attention.is_query_gated,
                "n_heads": attention.n_heads,
                "n_kv_heads": attention.n_kv_heads,
                "head_dim": attention.head_dim,
                "query_out_dims": attention.query_out_dims,
                "q_proj_out_dims": attention.attention_shape.q_proj_out_dims,
            }
        )
    return details


def linear_wrapped_details(model: Any, layers: list[int]) -> list[dict[str, Any]]:
    details = []
    for layer_index in layers:
        attention = get_layers(model)[layer_index].linear_attn
        details.append(
            {
                "layer": layer_index,
                "attention_class": type(attention.base).__name__,
                "hidden_size": attention.hidden_size,
                "rank": attention.delta_config.rank,
            }
        )
    return details


def get_layers(model: Any) -> Any:
    return getattr(getattr(model, "model", model), "layers")


def infer_hidden_size(attention: Any) -> int:
    from delta_mem_sidecar.mlx_delta_attention import _linear_output_dims

    hidden_size = _linear_output_dims(getattr(attention, "o_proj", None))
    if hidden_size is None:
        raise RuntimeError(f"Could not infer hidden size for {type(attention).__name__}")
    return int(hidden_size)


def infer_effective_query_width(attention: Any) -> int:
    from delta_mem_sidecar.mlx_delta_attention import _linear_output_dims

    q_width = _linear_output_dims(getattr(attention, "q_proj", None))
    if q_width is None:
        raise RuntimeError(f"Could not infer query width for {type(attention).__name__}")
    if "qwen3next" in type(attention).__name__.lower():
        return int(q_width) // 2
    return int(q_width)


def require_gpu(mx: Any) -> None:
    if "gpu" not in repr(mx.default_device()).lower():
        raise RuntimeError(f"Expected MLX GPU default device, got {mx.default_device()!r}")


def memory_snapshot(mx: Any) -> dict[str, Any]:
    snapshot = {
        "rss_max_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2),
    }
    metal = getattr(mx, "metal", None)
    if metal is not None:
        for name in ("get_active_memory", "get_peak_memory", "get_cache_memory"):
            fn = getattr(metal, name, None)
            if callable(fn):
                snapshot[name.removeprefix("get_") + "_mb"] = round(fn() / 1024**2, 2)
    return snapshot


if __name__ == "__main__":
    raise SystemExit(main())
