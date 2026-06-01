#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
TRAINER="${TRAINER:-$REPO_ROOT/scripts/train_mlx_qwen35_delta_mem.py}"
MODEL="${MODEL:-mlx-community/Qwen3.5-9B-MLX-4bit}"
TRAIN_FILE="${TRAIN_FILE:-$REPO_ROOT/examples/tiny_memory_train.json}"
PYTHONPATH_VALUE="${PYTHONPATH_VALUE:-$REPO_ROOT/src}"
TARGET_ALL8="3,7,11,15,19,23,27,31"

LAYER31_1_OUT="${LAYER31_1_OUT:-$REPO_ROOT/outputs/qwen35-layer31-1step}"
LAYER31_10_OUT="${LAYER31_10_OUT:-$REPO_ROOT/outputs/qwen35-layer31-10step}"
ALL8_1_OUT="${ALL8_1_OUT:-$REPO_ROOT/outputs/qwen35-fullattn-1step}"
ALL8_10_OUT="${ALL8_10_OUT:-$REPO_ROOT/outputs/qwen35-fullattn-10step}"

usage() {
  cat <<'EOF'
usage: phase4_qwen35_native_smokes.sh [print|preflight|layer31-1step|layer31-10step|all8-1step|all8-10step|validate DIR LAYERS COUNT]

Runs Phase 4 native-MLX Qwen3.5 smokes. Training modes refuse to overwrite
existing output directories.

Implementation note: all-8 training requires the trainer to keep the frozen
Qwen3.5 model in training mode so GatedDeltaNet uses the differentiable ops path
instead of the eval-mode custom Metal kernel.
EOF
}

require_absent_dir() {
  local out="$1"
  if [[ -e "$out" ]]; then
    echo "Refusing to overwrite existing output dir: $out" >&2
    exit 1
  fi
}

run_trainer() {
  local out="$1"
  local layers="$2"
  local steps="$3"
  require_absent_dir "$out"
  cd "$REPO_ROOT"
  PYTHONPATH="$PYTHONPATH_VALUE" TOKENIZERS_PARALLELISM=false "$PYTHON" "$TRAINER" \
    --model "$MODEL" \
    --train-file "$TRAIN_FILE" \
    --output-dir "$out" \
    --target-layers "$layers" \
    --rank 2 \
    --alpha 4.0 \
    --max-length 32 \
    --max-write-length 64 \
    --max-steps "$steps" \
    --learning-rate 2e-4 \
    --seed 17 \
    --beta-bias-init -1.5 \
    --validate-export
}

validate_native_export() {
  local out="$1"
  local expected_layers="$2"
  local expected_count="$3"
  cd "$REPO_ROOT"
  OUT="$out" EXPECTED_LAYERS="$expected_layers" EXPECTED_COUNT="$expected_count" \
    PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

import mlx.core as mx

from delta_mem_sidecar.mlx_delta_adapter import load_mlx_delta_adapter

out = Path(os.environ["OUT"])
expected_layers = [int(x) for x in os.environ["EXPECTED_LAYERS"].split(",") if x]
expected_count = int(os.environ["EXPECTED_COUNT"])
adapter = load_mlx_delta_adapter(out)
actual_layers = sorted(adapter.weights_by_layer)
if actual_layers != expected_layers:
    raise SystemExit(f"layer mismatch: {actual_layers} != {expected_layers}")
arrays = dict(mx.load(str(out / "delta_mem_adapter_mlx.npz")))
if len(arrays) != expected_count:
    raise SystemExit(f"tensor count mismatch: {len(arrays)} != {expected_count}")
for key, value in arrays.items():
    if not bool(mx.all(mx.isfinite(value)).item()):
        raise SystemExit(f"non-finite export tensor: {key}")
summary = json.loads((out / "training_summary.json").read_text())
if summary.get("adapter_changed") is not True:
    raise SystemExit("adapter_changed was not true")
print(f"native export validated: {out}")
PY
}

mode="${1:-print}"
case "$mode" in
  print)
    usage
    ;;
  preflight)
    cd "$REPO_ROOT"
    PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON" -m py_compile "$TRAINER"
    PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON" "$TRAINER" --help
    PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON" "$REPO_ROOT/scripts/probe_mlx_env.py"
    df -h "$REPO_ROOT" "${TMPDIR:-/tmp}"
    vm_stat
    ;;
  layer31-1step)
    run_trainer "$LAYER31_1_OUT" "31" "1"
    validate_native_export "$LAYER31_1_OUT" "31" "7"
    ;;
  layer31-10step)
    run_trainer "$LAYER31_10_OUT" "31" "10"
    validate_native_export "$LAYER31_10_OUT" "31" "7"
    ;;
  all8-1step)
    run_trainer "$ALL8_1_OUT" "$TARGET_ALL8" "1"
    validate_native_export "$ALL8_1_OUT" "$TARGET_ALL8" "56"
    ;;
  all8-10step)
    run_trainer "$ALL8_10_OUT" "$TARGET_ALL8" "10"
    validate_native_export "$ALL8_10_OUT" "$TARGET_ALL8" "56"
    ;;
  validate)
    validate_native_export "${2:?DIR required}" "${3:?LAYERS required}" "${4:?COUNT required}"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
