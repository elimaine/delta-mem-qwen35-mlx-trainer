# Qwen3.5 training runbook

## Known-good mechanical recipe

Start small:

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
  --learning-rate 2e-4 \
  --seed 17 \
  --validate-export
```

Scale only after the one-layer smoke passes:

```sh
PYTHONPATH=src TOKENIZERS_PARALLELISM=false python \
  scripts/train_mlx_qwen35_delta_mem.py \
  --train-file examples/tiny_memory_train.json \
  --output-dir outputs/qwen35-fullattn-10step \
  --target-layers 3,7,11,15,19,23,27,31 \
  --linear-layers none \
  --rank 2 \
  --alpha 4.0 \
  --max-length 32 \
  --max-write-length 64 \
  --max-steps 10 \
  --compile-step \
  --checkpoint-every 5 \
  --validate-export
```

## Layer map

Qwen3.5 9B has 32 layers. Full-attention layers are:

```text
3,7,11,15,19,23,27,31
```

The other layers are GatedDeltaNet layers. The trainer can attach output-only
delta adapters to those layers with `--linear-layers`.

## Important caveats

- `model.train()` is required even though the base model is frozen. MLX-LM uses
  an inference custom kernel for Qwen3.5 GatedDeltaNet in eval mode, and that
  kernel has no VJP.
- `--grad-checkpoint` is intentionally rejected. A naive MLX-LM layer checkpoint
  wrapper can produce finite loss while silently dropping external adapter
  updates.
- Checkpoint directories contain adapter artifacts for evaluation, not optimizer
  state for exact resume.
