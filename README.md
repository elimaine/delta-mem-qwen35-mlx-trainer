# delta-mem Qwen3.5 MLX trainer

Public-facing research scaffold for training small delta-memory adapters on
`mlx-community/Qwen3.5-9B-MLX-4bit` with Apple Silicon MLX.

This repo is a focused spin-off of a larger local research workspace. It keeps
the reusable native-MLX trainer, runtime adapter loader, and no-heavy unit tests,
while excluding local planning logs, model checkpoints, benchmark outputs, and
machine-specific paths.

## Status

Mechanically verified:

- Qwen3.5 MLX full-attention delta adapters can be initialized, trained, exported,
  and loaded back through the runtime adapter loader.
- Full-attention Qwen3.5 query-gate handling is implemented.
- Optional output-only wrappers for Qwen3.5 GatedDeltaNet layers are implemented.
- The trainer can export `delta_mem_adapter_mlx.npz`, `delta_mem_config.json`,
  `training_summary.json`, and `benchmark.json`.

Not yet proven:

- The exported adapters improve held-out memory quality.
- Training can resume an exact optimizer trajectory from a checkpoint.
- The recipe is tuned beyond small smoke/pilot runs.

## Setup

```sh
python3.11 -m venv .venv
. .venv/bin/activate
pip install -e ".[test]"
```

On Apple Silicon, confirm MLX and a small toy model load:

```sh
python scripts/probe_mlx_env.py
```

## One-step smoke

This uses the tiny in-repo fixture and trains only the last Qwen3.5
full-attention layer.

```sh
PYTHONPATH=src TOKENIZERS_PARALLELISM=false python \
  scripts/train_mlx_qwen35_delta_mem.py \
  --train-file examples/tiny_memory_train.json \
  --output-dir outputs/qwen35-layer31-1step \
  --target-layers 31 \
  --linear-layers none \
  --rank 2 \
  --alpha 4.0 \
  --max-length 32 \
  --max-write-length 32 \
  --max-steps 1 \
  --validate-export
```

All Qwen3.5 full-attention layers are:

```text
3,7,11,15,19,23,27,31
```

All Qwen3.5 GatedDeltaNet output layers can be selected by passing the
non-full-attention layer ids to `--linear-layers`.

## Useful checks

No-heavy tests:

```sh
PYTHONPATH=src pytest -q
```

Real-model zero-delta parity:

```sh
PYTHONPATH=src python scripts/qwen35_zero_delta_parity.py \
  --layers all \
  --linear-layers all
```

## Checkpoints

`--checkpoint-every N` writes adapter artifacts under
`<output-dir>/checkpoints/step-N`. These checkpoints are useful for evaluation
and serving tests. They are not exact training-resume checkpoints because
optimizer state and RNG state are not reloaded by the trainer yet.

## License and provenance

No model weights or trained adapters are included. Check the licenses for the
base model and any dataset you train on before publishing derived adapters.
