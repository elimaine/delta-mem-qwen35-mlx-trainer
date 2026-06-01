#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_TOY_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
DEFAULT_QWEN35_MODEL = "mlx-community/Qwen3.5-9B-MLX-4bit"


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe local MLX/MLX-LM environment and model load support.")
    parser.add_argument("--toy-model", default=DEFAULT_TOY_MODEL)
    parser.add_argument("--qwen35-model", default=DEFAULT_QWEN35_MODEL)
    parser.add_argument("--skip-toy-load", action="store_true")
    parser.add_argument("--skip-hf-check", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report: dict[str, Any] = {
        "python": {
            "executable": sys.executable,
            "version": sys.version.replace("\n", " "),
            "prefix": sys.prefix,
        },
        "platform": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "macos": platform.mac_ver()[0],
        },
        "disk": _disk_headroom(Path.cwd()),
        "packages": {
            "mlx": _package_report("mlx"),
            "mlx-lm": _package_report("mlx-lm"),
            "huggingface-hub": _package_report("huggingface-hub"),
        },
        "mlx": {},
        "toy_model": {"model_id": args.toy_model, "attempted_load": False},
        "qwen35_model": {"model_id": args.qwen35_model},
    }

    mx = _import_optional("mlx.core")
    if mx is not None:
        try:
            report["mlx"]["default_device"] = repr(mx.default_device())
        except Exception as exc:  # noqa: BLE001
            report["mlx"]["default_device_error"] = f"{type(exc).__name__}: {exc}"

    mlx_lm = _import_optional("mlx_lm")
    if not args.skip_hf_check:
        report["qwen35_model"]["hub"] = _model_info(args.qwen35_model)

    if mlx_lm is not None and not args.skip_toy_load:
        report["toy_model"]["attempted_load"] = True
        start = time.time()
        try:
            model, tokenizer = mlx_lm.load(args.toy_model)
        except Exception as exc:  # noqa: BLE001
            report["toy_model"]["load_ok"] = False
            report["toy_model"]["error"] = f"{type(exc).__name__}: {exc}"
        else:
            report["toy_model"]["load_ok"] = True
            report["toy_model"]["load_seconds"] = round(time.time() - start, 3)
            report["toy_model"]["summary"] = _summarize_model(model)
            report["toy_model"]["tokenizer_class"] = type(tokenizer).__name__

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_format_report(report))
    return 0


def _import_optional(name: str) -> Any | None:
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def _package_report(name: str) -> dict[str, Any]:
    try:
        return {"installed": True, "version": metadata.version(name)}
    except metadata.PackageNotFoundError:
        return {"installed": False, "version": None}


def _disk_headroom(path: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    gib = 1024**3
    return {
        "path": str(path),
        "total_gib": round(usage.total / gib, 2),
        "used_gib": round(usage.used / gib, 2),
        "free_gib": round(usage.free / gib, 2),
    }


def _model_info(model_id: str) -> dict[str, Any]:
    hub = _import_optional("huggingface_hub")
    if hub is None:
        return {"checked": False, "error": "huggingface_hub is not importable"}
    try:
        info = hub.model_info(model_id, files_metadata=True)
    except Exception as exc:  # noqa: BLE001
        return {"checked": True, "available": False, "error": f"{type(exc).__name__}: {exc}"}
    siblings = list(getattr(info, "siblings", []) or [])
    total_size = sum(size for item in siblings if isinstance((size := getattr(item, "size", None)), int))
    return {
        "checked": True,
        "available": True,
        "sha": getattr(info, "sha", None),
        "library_name": getattr(info, "library_name", None),
        "tags": list(getattr(info, "tags", []) or []),
        "sibling_count": len(siblings),
        "reported_size_gib": round(total_size / 1024**3, 2) if total_size else None,
    }


def _summarize_model(model: Any) -> dict[str, Any]:
    layers = _get_attr(model, "model.layers") or _get_attr(model, "layers")
    layer_list = list(layers) if layers is not None else []
    config = getattr(model, "args", None) or getattr(model, "config", None)
    summary: dict[str, Any] = {
        "model_class": type(model).__name__,
        "config_class": type(config).__name__ if config is not None else None,
        "model_type": getattr(config, "model_type", None),
        "num_layers": len(layer_list),
    }
    if layer_list:
        summary["first_layer_class"] = type(layer_list[0]).__name__
        summary["last_layer_class"] = type(layer_list[-1]).__name__
        summary["first_attention_class"] = type(getattr(layer_list[0], "self_attn", None)).__name__
        quantized = []
        for path in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"):
            module = _get_attr(layer_list[0], path)
            if module is None:
                continue
            if "Quantized" in type(module).__name__ or hasattr(module, "bits"):
                quantized.append(
                    {
                        "path": f"layer0.{path}",
                        "class": type(module).__name__,
                        "bits": getattr(module, "bits", None),
                        "group_size": getattr(module, "group_size", None),
                        "mode": getattr(module, "mode", None),
                    }
                )
        summary["quantized_modules_sample"] = quantized
    return summary


def _get_attr(obj: Any, dotted: str) -> Any:
    current = obj
    for part in dotted.split("."):
        current = getattr(current, part, None)
        if current is None:
            return None
    return current


def _format_report(report: dict[str, Any]) -> str:
    lines = [
        f"python: {report['python']['executable']}",
        f"python_version: {report['python']['version'].split()[0]}",
        f"disk_free_gib: {report['disk']['free_gib']}",
    ]
    for name, pkg in report["packages"].items():
        lines.append(f"{name}: {pkg['version'] if pkg['installed'] else 'missing'}")
    if "default_device" in report["mlx"]:
        lines.append(f"mlx_default_device: {report['mlx']['default_device']}")
    toy = report["toy_model"]
    if toy.get("attempted_load"):
        lines.append(f"toy_load_ok: {toy.get('load_ok')}")
        if toy.get("load_ok"):
            summary = toy["summary"]
            lines.append(f"toy_model_type: {summary.get('model_type')}")
            lines.append(f"toy_layers: {summary.get('num_layers')}")
            lines.append(f"toy_quantized_sample: {summary.get('quantized_modules_sample')}")
        else:
            lines.append(f"toy_error: {toy.get('error')}")
    qwen35 = report["qwen35_model"].get("hub", {})
    if qwen35:
        lines.append(f"qwen35_available: {qwen35.get('available')}")
        lines.append(f"qwen35_library: {qwen35.get('library_name')}")
        lines.append(f"qwen35_reported_size_gib: {qwen35.get('reported_size_gib')}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
