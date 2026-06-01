#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "mlx-community/Qwen3.5-9B-MLX-4bit"
DEFAULT_ADAPTER = "outputs/qwen35-layer31-1step"
FULL_ATTENTION_LAYERS = (3, 7, 11, 15, 19, 23, 27, 31)


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke Qwen3.5 MLX delta adapter wrapping.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter-dir", default=DEFAULT_ADAPTER)
    parser.add_argument("--layers", default="31", help="Comma-separated layer ids or 'all'.")
    parser.add_argument("--prompt", default="Remember: Delta is local memory. What is Delta?")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    from delta_mem_sidecar.mlx_delta_adapter import load_mlx_delta_adapter
    from delta_mem_sidecar.mlx_delta_attention import (
        iter_mlx_delta_attention_modules,
        reset_mlx_delta_states,
        wrap_qwen_attention_layers,
    )
    from delta_mem_sidecar.mlx_runtime import _validate_adapter_compatible

    _require_gpu(mx)
    layers = _parse_layers(args.layers)
    report: dict[str, Any] = {
        "model": args.model,
        "adapter_dir": args.adapter_dir,
        "layers": layers,
        "timing_seconds": {},
        "memory": {"initial": _memory_snapshot(mx)},
    }

    start = time.perf_counter()
    model, tokenizer = load(args.model)
    report["timing_seconds"]["load_model"] = round(time.perf_counter() - start, 3)
    report["memory"]["after_model_load"] = _memory_snapshot(mx)

    start = time.perf_counter()
    adapter = load_mlx_delta_adapter(args.adapter_dir)
    report["timing_seconds"]["load_adapter"] = round(time.perf_counter() - start, 3)
    selected_weights = {layer: adapter.weights_by_layer[layer] for layer in layers}

    _validate_adapter_compatible(model, selected_weights)
    before = [_projection_snapshot(model, layer) for layer in layers]
    wrapped = wrap_qwen_attention_layers(model, weights_by_layer=selected_weights, config=adapter.config)
    after = [_projection_snapshot(model, layer) for layer in layers]
    report["wrapped"] = wrapped
    report["projection_before"] = before
    report["projection_after"] = after

    wrapped_details = []
    for layer, attention in iter_mlx_delta_attention_modules(model):
        wrapped_details.append(
            {
                "layer": layer,
                "is_query_gated": attention.is_query_gated,
                "n_heads": attention.n_heads,
                "n_kv_heads": attention.n_kv_heads,
                "head_dim": attention.head_dim,
                "query_out_dims": attention.query_out_dims,
                "q_proj_out_dims": attention.attention_shape.q_proj_out_dims,
                "q_proj_bits": getattr(attention.q_proj, "bits", None),
            }
        )
    report["wrapped_details"] = wrapped_details

    tokens = mx.array([tokenizer.encode(args.prompt)])
    reset_mlx_delta_states(model)
    start = time.perf_counter()
    logits = model(tokens)
    mx.eval(logits)
    report["timing_seconds"]["forward"] = round(time.perf_counter() - start, 3)
    report["logits_shape"] = list(logits.shape)
    report["logits_finite"] = bool(mx.all(mx.isfinite(logits)).item())
    report["memory"]["after_forward"] = _memory_snapshot(mx)
    if not report["logits_finite"]:
        raise RuntimeError("Qwen3.5 MLX wrapped forward produced non-finite logits")

    output = json.dumps(report, indent=2, sort_keys=True)
    if args.json_out:
        args.json_out.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0


def _parse_layers(raw: str) -> list[int]:
    if raw == "all":
        return list(FULL_ATTENTION_LAYERS)
    layers = [int(part) for part in raw.split(",") if part]
    for layer in layers:
        if layer not in FULL_ATTENTION_LAYERS:
            raise ValueError(f"Layer {layer} is not a Qwen3.5 full-attention layer")
    return layers


def _projection_snapshot(model: Any, layer: int) -> dict[str, Any]:
    attention = model.layers[layer].self_attn
    return {
        "layer": layer,
        "attention_class": type(attention).__name__,
        "q_class": type(attention.q_proj).__name__,
        "q_bits": getattr(attention.q_proj, "bits", None),
        "q_group_size": getattr(attention.q_proj, "group_size", None),
        "k_class": type(attention.k_proj).__name__,
        "v_class": type(attention.v_proj).__name__,
        "o_class": type(attention.o_proj).__name__,
    }


def _require_gpu(mx: Any) -> None:
    if "gpu" not in repr(mx.default_device()).lower():
        raise RuntimeError(f"Expected MLX GPU default device, got {mx.default_device()!r}")


def _memory_snapshot(mx: Any) -> dict[str, Any]:
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
