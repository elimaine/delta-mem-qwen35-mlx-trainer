#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import resource
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, Iterable

from delta_mem_sidecar.mlx_delta_attention import (
    MlxDeltaAttentionConfig,
    MlxDeltaAttentionWeights,
    MlxDeltaOutputWeights,
    _linear_output_dims,
    _normalize_memory_projection,
    delta_affine_scan,
    infer_mlx_attention_shape,
)


DEFAULT_MODEL = "mlx-community/Qwen3.5-9B-MLX-4bit"
DEFAULT_TRAIN_FILE = "examples/tiny_memory_train.json"
DEFAULT_TARGET_LAYERS = (3, 7, 11, 15, 19, 23, 27, 31)
ADAPTER_KEYS = (
    "memory_q_proj",
    "memory_k_proj",
    "memory_v_proj",
    "delta_q_proj",
    "delta_o_proj",
    "beta_proj",
    "beta_bias",
)
OUTPUT_ADAPTER_KEYS = (
    "memory_q_proj",
    "memory_k_proj",
    "memory_v_proj",
    "delta_o_proj",
    "beta_proj",
    "beta_bias",
)
IGNORE_INDEX = -100
MAX_CHAT_TEMPLATE_SUFFIX_ROLLBACK_TOKENS = 16


@dataclass(frozen=True)
class Batch:
    input_ids: Any
    labels: Any
    write_input_ids: Any | None
    fallback_input_ids: Any | None
    fallback_labels: Any | None
    shape_signature: dict[str, Any]


def main() -> None:
    args = parse_args()
    target_layers = parse_target_layers(args.target_layers)
    target_linear_layers = parse_target_layers(args.linear_layers)
    if args.log_every < 0:
        raise ValueError("--log-every must be >= 0")
    if args.checkpoint_every < 0:
        raise ValueError("--checkpoint-every must be >= 0")
    if not target_layers and not target_linear_layers:
        raise ValueError("At least one of --target-layers or --linear-layers must select layers")
    if args.output_dir.exists() and not args.overwrite:
        raise RuntimeError(f"Refusing to overwrite existing output dir: {args.output_dir}")

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx_lm import load

    if not args.allow_cpu:
        require_mlx_gpu(mx)
    mx.random.seed(args.seed)

    report: dict[str, Any] = {
        "model": args.model,
        "training_mode": args.training_mode,
        "episode_recent_messages": args.episode_recent_messages,
        "target_layers": list(target_layers),
        "target_linear_layers": list(target_linear_layers),
        "rank": args.rank,
        "max_length": args.max_length,
        "max_write_length": args.max_write_length,
        "max_steps": args.max_steps,
        "log_every": args.log_every,
        "checkpoint_every": args.checkpoint_every,
        "memory": {"initial": memory_snapshot(mx)},
        "timing_seconds": {},
    }
    total_start = time.perf_counter()

    start = time.perf_counter()
    model, tokenizer = load(args.model)
    report["timing_seconds"]["load_model"] = time.perf_counter() - start
    report["memory"]["after_model_load"] = memory_snapshot(mx)
    freeze_base_model_contract(model)
    if hasattr(model, "train"):
        model.train()
    if args.grad_checkpoint:
        raise RuntimeError(
            "--grad-checkpoint is not supported yet for external delta-mem adapter training; "
            "the mlx-lm layer checkpoint pattern drops adapter updates in this trainer."
        )

    start = time.perf_counter()
    adapter = init_adapter_tree(
        model,
        target_layers=target_layers,
        target_linear_layers=target_linear_layers,
        rank=args.rank,
        beta_bias_init=args.beta_bias_init,
        nn=nn,
        mx=mx,
    )
    assert_optimizer_only_adapter_contract(model, adapter)
    report["timing_seconds"]["build_adapter"] = time.perf_counter() - start
    report["memory"]["after_adapter_init"] = memory_snapshot(mx)
    start = time.perf_counter()
    install_trainable_delta_attention(
        model,
        adapter,
        alpha=args.alpha,
        rank=args.rank,
        max_write_length=args.max_write_length,
    )
    report["timing_seconds"]["install_delta_attention"] = time.perf_counter() - start

    start = time.perf_counter()
    examples = load_chat_examples(args.train_file)
    if args.exclude_fallback_answers:
        examples = [
            example
            for example in examples
            if normalize_answer_text(example["messages"][-1].get("content", "")) != normalize_answer_text(args.fallback_answer)
        ]
        if not examples:
            raise ValueError("--exclude-fallback-answers removed every example")
    if args.shuffle_examples:
        random.Random(args.seed).shuffle(examples)
    if args.dataset_limit is not None:
        examples = examples[: args.dataset_limit]
    batches = [
        tokenize_training_example(
            example["messages"],
            tokenizer,
            training_mode=args.training_mode,
            max_length=args.max_length,
            max_write_length=args.max_write_length,
            episode_recent_messages=args.episode_recent_messages,
            fallback_answer=args.fallback_answer,
            mx=mx,
        )
        for example in examples
    ]
    report["timing_seconds"]["tokenize_dataset"] = time.perf_counter() - start
    report["shape_signatures"] = [batch.shape_signature for batch in batches]

    optimizer = optim.AdamW(learning_rate=args.learning_rate, weight_decay=0.0)
    initial_adapter = tree_copy(adapter.trainable_parameters(), mx)
    loss_history: list[float] = []
    step_times: list[dict[str, Any]] = []
    checkpoint_records: list[dict[str, Any]] = []
    current_step = {"value": 0}

    def loss_fn(
        trainable_adapter: Any,
        input_ids: Any,
        labels: Any,
        write_input_ids: Any | None = None,
        fallback_input_ids: Any | None = None,
        fallback_labels: Any | None = None,
    ) -> Any:
        if trainable_adapter is not adapter:
            raise RuntimeError("value_and_grad must update the installed adapter module")
        reset_trainable_delta_states(model)
        if write_input_ids is not None:
            set_trainable_delta_write_enabled(model, True)
            _ = model(write_input_ids)
            set_trainable_delta_write_enabled(model, False)
        else:
            set_trainable_delta_write_enabled(model, True)
        logits = model(input_ids)
        set_trainable_delta_write_enabled(model, True)
        keep_loss = combined_training_loss(
            logits,
            labels,
            nn,
            mx,
            answer_focus_weight=args.answer_focus_weight,
            answer_focus_tokens=args.answer_focus_tokens,
        )
        fallback_due = (
            args.fallback_contrast_weight > 0.0
            and args.fallback_contrast_every > 0
            and (current_step["value"] + 1) % args.fallback_contrast_every == 0
            and fallback_input_ids is not None
            and fallback_labels is not None
        )
        if fallback_due:
            fallback_logits = model(fallback_input_ids)
            fallback_loss = focused_answer_cross_entropy(
                fallback_logits,
                fallback_labels,
                nn,
                mx,
                answer_focus_tokens=args.answer_focus_tokens,
            )
            gold_focus_loss = focused_answer_cross_entropy(
                logits,
                labels,
                nn,
                mx,
                answer_focus_tokens=args.answer_focus_tokens,
            )
            fallback_gap = mx.clip(
                fallback_loss - gold_focus_loss,
                mx.array(-args.fallback_gap_clip, dtype=fallback_loss.dtype),
                mx.array(args.fallback_gap_clip, dtype=fallback_loss.dtype),
            )
            keep_loss = keep_loss + args.fallback_contrast_weight * pairwise_margin_objective(
                fallback_gap,
                args.fallback_margin,
                mx,
            )
        contrast_due = args.memory_contrast_every > 0 and (current_step["value"] + 1) % args.memory_contrast_every == 0
        if args.memory_contrast_weight <= 0.0 or not contrast_due:
            return keep_loss
        reset_trainable_delta_states(model)
        set_trainable_delta_write_enabled(model, False)
        reset_logits = model(input_ids)
        set_trainable_delta_write_enabled(model, True)
        reset_loss = combined_training_loss(
            reset_logits,
            labels,
            nn,
            mx,
            answer_focus_weight=args.answer_focus_weight,
            answer_focus_tokens=args.answer_focus_tokens,
        )
        gap = reset_loss - keep_loss
        margin_loss = memory_margin_objective(gap, args.memory_margin, mx)
        loss = keep_loss + args.memory_contrast_weight * margin_loss
        return loss

    value_and_grad = nn.value_and_grad(adapter, loss_fn)
    compiled = bool(args.compile_step)

    def uncompiled_train_step(
        input_ids: Any,
        labels: Any,
        write_input_ids: Any | None,
        fallback_input_ids: Any | None,
        fallback_labels: Any | None,
    ) -> tuple[Any, Any]:
        loss, grads = value_and_grad(adapter, input_ids, labels, write_input_ids, fallback_input_ids, fallback_labels)
        optimizer.update(adapter, grads)
        mx.eval(adapter.parameters(), optimizer.state, loss)
        return loss, grads

    train_step = uncompiled_train_step
    if compiled:
        state = [adapter.state, optimizer.state, mx.random.state]

        @partial(mx.compile, inputs=state, outputs=state)
        def compiled_train_step(
            input_ids: Any,
            labels: Any,
            write_input_ids: Any | None,
            fallback_input_ids: Any | None,
            fallback_labels: Any | None,
        ) -> tuple[Any, Any]:
            loss, grads = value_and_grad(adapter, input_ids, labels, write_input_ids, fallback_input_ids, fallback_labels)
            optimizer.update(adapter, grads)
            return loss, grads

        train_step = compiled_train_step

    train_start = time.perf_counter()
    for step in range(args.max_steps):
        current_step["value"] = step
        batch = batches[step % len(batches)]
        step_start = time.perf_counter()
        loss, grads = train_step(
            batch.input_ids,
            batch.labels,
            batch.write_input_ids,
            batch.fallback_input_ids,
            batch.fallback_labels,
        )
        if compiled:
            mx.eval(adapter.parameters(), optimizer.state, loss)
        if args.clear_cache_every and (step + 1) % args.clear_cache_every == 0:
            mx.clear_cache()
        require_finite("loss", loss, mx)
        require_finite_tree("gradients", grads, mx)
        require_finite_tree("adapter_after_update", adapter.trainable_parameters(), mx)
        loss_value = float(loss.item())
        loss_history.append(loss_value)
        step_time = time.perf_counter() - step_start
        step_times.append(
            {
                "step": step + 1,
                "loss": loss_value,
                "wall_seconds": step_time,
                "steps_per_second": 1.0 / step_time if step_time > 0 else None,
            }
        )
        if should_log_step(step + 1, args.max_steps, args.log_every):
            print(json.dumps({"step": step + 1, "loss": loss_value}, sort_keys=True))
        if should_checkpoint_step(step + 1, args.max_steps, args.checkpoint_every):
            checkpoint_dir = args.output_dir / "checkpoints" / f"step-{step + 1}"
            checkpoint_start = time.perf_counter()
            checkpoint_summary = build_training_summary(
                args=args,
                target_layers=target_layers,
                target_linear_layers=target_linear_layers,
                compiled=compiled,
                loss_history=loss_history,
                step_times=step_times,
                changed=None,
                train_elapsed=time.perf_counter() - train_start,
                report=report,
            )
            write_training_artifacts(
                output_dir=checkpoint_dir,
                adapter=adapter,
                args=args,
                target_layers=target_layers,
                target_linear_layers=target_linear_layers,
                summary=checkpoint_summary,
                report={
                    **report,
                    "checkpoint_step": step + 1,
                    "checkpoint_created_at": utc_now_iso(),
                },
                mx=mx,
                validate=args.validate_export,
            )
            elapsed = time.perf_counter() - checkpoint_start
            checkpoint_records.append(
                {
                    "step": step + 1,
                    "path": str(checkpoint_dir),
                    "elapsed_s": elapsed,
                }
            )
            print(json.dumps({"checkpoint": str(checkpoint_dir), "step": step + 1}, sort_keys=True))

    train_elapsed = time.perf_counter() - train_start
    changed = adapter_tree_changed(initial_adapter, adapter.trainable_parameters(), mx)
    if not changed:
        raise RuntimeError("adapter tensors did not change; optimizer/update path is not training the adapter")
    report["timing_seconds"]["train_wall"] = train_elapsed
    report["memory"]["after_train"] = memory_snapshot(mx)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    summary = build_training_summary(
        args=args,
        target_layers=target_layers,
        target_linear_layers=target_linear_layers,
        compiled=compiled,
        loss_history=loss_history,
        step_times=step_times,
        changed=changed,
        train_elapsed=train_elapsed,
        report=report,
        checkpoint_records=checkpoint_records,
    )
    write_training_artifacts(
        output_dir=args.output_dir,
        adapter=adapter,
        args=args,
        target_layers=target_layers,
        target_linear_layers=target_linear_layers,
        summary=summary,
        report=report,
        mx=mx,
        validate=False,
    )
    report["timing_seconds"]["export"] = time.perf_counter() - start
    report["memory"]["after_export"] = memory_snapshot(mx)

    start = time.perf_counter()
    validate_export_files(args.output_dir, adapter, target_layers, target_linear_layers, args.rank, mx)
    if args.validate_export:
        validate_with_sidecar_loader(args.output_dir)
    report["timing_seconds"]["validation"] = time.perf_counter() - start
    report["timing_seconds"]["total_wall"] = time.perf_counter() - total_start
    report["memory"]["after_validation"] = memory_snapshot(mx)
    report["artifact_dir"] = str(args.output_dir)
    report["written_at"] = utc_now_iso()
    (args.output_dir / "benchmark.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Native MLX Qwen3.5 delta-memory adapter trainer.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--train-file", type=Path, default=Path(DEFAULT_TRAIN_FILE))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-layers", default=",".join(map(str, DEFAULT_TARGET_LAYERS)))
    parser.add_argument(
        "--linear-layers",
        default="none",
        help="Comma-separated Qwen3.5 GatedDeltaNet layer ids to train with output-only delta-mem, or 'none'.",
    )
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--max-write-length", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument(
        "--training-mode",
        choices=("dialogue", "episode"),
        default="dialogue",
        help="dialogue trains one visible chat; episode writes earlier turns into delta state and reads the recent query/answer.",
    )
    parser.add_argument("--episode-recent-messages", type=int, default=1)
    parser.add_argument("--dataset-limit", type=int)
    parser.add_argument("--shuffle-examples", action="store_true")
    parser.add_argument("--memory-contrast-weight", type=float, default=0.0)
    parser.add_argument("--memory-contrast-every", type=int, default=1, help="Apply contrast objective every N steps. Use 0 to disable.")
    parser.add_argument("--memory-margin", type=float, default=0.01)
    parser.add_argument(
        "--answer-focus-weight",
        type=float,
        default=0.0,
        help="Extra loss weight on the first supervised answer tokens. Useful for short-span answer training.",
    )
    parser.add_argument(
        "--answer-focus-tokens",
        type=int,
        default=4,
        help="Number of leading supervised answer tokens to emphasize when --answer-focus-weight > 0.",
    )
    parser.add_argument("--fallback-answer", default="No information available.")
    parser.add_argument(
        "--exclude-fallback-answers",
        action="store_true",
        help="Drop examples whose final assistant answer is the fallback answer.",
    )
    parser.add_argument(
        "--fallback-contrast-weight",
        type=float,
        default=0.0,
        help="Extra memory-on pairwise loss requiring gold answer CE to beat fallback answer CE.",
    )
    parser.add_argument(
        "--fallback-contrast-every",
        type=int,
        default=1,
        help="Apply fallback-vs-gold contrast every N steps. Use 0 to disable.",
    )
    parser.add_argument("--fallback-margin", type=float, default=0.2)
    parser.add_argument("--fallback-gap-clip", type=float, default=10.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--log-every", type=int, default=1, help="Print per-step JSON every N steps. Use 0 to print only the final summary.")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Export adapter/config/partial summaries every N steps under <output-dir>/checkpoints/step-N. Use 0 to disable.",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--beta-bias-init", type=float, default=-1.5)
    parser.add_argument("--validate-export", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--compile-step", action="store_true")
    parser.add_argument(
        "--grad-checkpoint",
        action="store_true",
        help="Reserved for future adapter-aware checkpointing; currently rejected to avoid silent no-op adapter updates.",
    )
    parser.add_argument("--clear-cache-every", type=int, default=0, help="Call mx.clear_cache() every N steps. Use 0 to disable.")
    parser.add_argument("--allow-cpu", action="store_true", help="Only for parser/unit smoke; real training must use MLX GPU.")
    return parser.parse_args()


def parse_target_layers(raw: str) -> tuple[int, ...]:
    if raw.strip() in {"", "off", "none"}:
        return ()
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def should_log_step(step: int, max_steps: int, log_every: int) -> bool:
    if log_every < 0:
        raise ValueError("--log-every must be >= 0")
    if log_every == 0:
        return step == max_steps
    return step == 1 or step == max_steps or step % log_every == 0


def should_checkpoint_step(step: int, max_steps: int, checkpoint_every: int) -> bool:
    if checkpoint_every < 0:
        raise ValueError("--checkpoint-every must be >= 0")
    if checkpoint_every == 0:
        return False
    return step < max_steps and step % checkpoint_every == 0


def require_mlx_gpu(mx: Any) -> None:
    if "gpu" not in repr(mx.default_device()).lower():
        raise RuntimeError(f"MLX default device is {mx.default_device()!r}, expected GPU")


def install_gradient_checkpointing(model: Any) -> None:
    import mlx.core as mx

    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if not layers:
        raise RuntimeError("Cannot install gradient checkpointing: model has no layers")
    layer_type = type(layers[0])
    if getattr(layer_type, "_delta_mem_checkpointed", False):
        return
    original_call = layer_type.__call__

    def checkpointed_call(layer: Any, *args: Any, **kwargs: Any) -> Any:
        def inner_call(params: Any, *inner_args: Any, **inner_kwargs: Any) -> Any:
            layer.update(params)
            return original_call(layer, *inner_args, **inner_kwargs)

        return mx.checkpoint(inner_call)(layer.trainable_parameters(), *args, **kwargs)

    layer_type.__call__ = checkpointed_call
    layer_type._delta_mem_checkpointed = True


def load_chat_examples(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        examples = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        examples = json.loads(text)
    if not isinstance(examples, list) or not examples:
        raise ValueError(f"training file must be a non-empty list: {path}")
    for index, example in enumerate(examples):
        messages = example.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"example {index} has no messages list")
        if messages[-1].get("role") != "assistant":
            raise ValueError(f"example {index} must end with the final assistant answer")
    return examples


def tokenize_training_example(
    messages: list[dict[str, str]],
    tokenizer: Any,
    *,
    training_mode: str,
    max_length: int,
    max_write_length: int,
    episode_recent_messages: int,
    fallback_answer: str,
    mx: Any,
) -> Batch:
    if training_mode == "dialogue":
        return tokenize_final_assistant_only(messages, tokenizer, max_length, max_write_length, fallback_answer, mx)
    if training_mode == "episode":
        return tokenize_episode_final_assistant_only(
            messages,
            tokenizer,
            max_length=max_length,
            max_write_length=max_write_length,
            episode_recent_messages=episode_recent_messages,
            fallback_answer=fallback_answer,
            mx=mx,
        )
    raise ValueError(f"Unsupported training mode: {training_mode}")


def tokenize_final_assistant_only(
    messages: list[dict[str, str]],
    tokenizer: Any,
    max_length: int,
    max_write_length: int,
    fallback_answer: str,
    mx: Any,
) -> Batch:
    tokenized = tokenize_messages_for_sft(
        tokenizer,
        messages,
        max_length=max_length,
        assistant_loss_mode="final_assistant_only",
    )
    input_ids = tokenized["input_ids"]
    labels = tokenized["labels"]
    if len(input_ids) < 2:
        raise ValueError("tokenized example is too short for next-token loss")
    if not any(label != IGNORE_INDEX for label in labels[1:]):
        raise ValueError("final assistant label span is empty after shifting/truncation")
    pad_token_id = int(getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None) or 0)
    padded_ids = input_ids + [pad_token_id] * (max_length - len(input_ids))
    padded_labels = labels + [IGNORE_INDEX] * (max_length - len(labels))
    return Batch(
        input_ids=mx.array([padded_ids], dtype=mx.int32),
        labels=mx.array([padded_labels], dtype=mx.int32),
        write_input_ids=None,
        fallback_input_ids=None,
        fallback_labels=None,
        shape_signature={
            "mode": "dialogue",
            "input_ids": [1, max_length],
            "labels": [1, max_length],
            "max_write_length": max_write_length,
        },
    )


def tokenize_episode_final_assistant_only(
    messages: list[dict[str, str]],
    tokenizer: Any,
    *,
    max_length: int,
    max_write_length: int,
    episode_recent_messages: int,
    fallback_answer: str,
    mx: Any,
) -> Batch:
    if episode_recent_messages < 0:
        raise ValueError("--episode-recent-messages must be >= 0")
    target_index = select_supervised_assistant_indices(messages, "final_assistant_only")[-1]
    prefix_messages = [dict(message) for message in messages[:target_index]]
    system_prefix, non_system_prefix = split_system_prefix(prefix_messages)
    if episode_recent_messages == 0:
        visible_non_system = []
        write_non_system = non_system_prefix
    else:
        visible_non_system = non_system_prefix[-episode_recent_messages:]
        write_non_system = non_system_prefix[:-episode_recent_messages]

    write_messages = system_prefix + write_non_system if write_non_system else []
    write_input_ids = truncate_token_sequence(tokenize_chat_messages(tokenizer, write_messages), max_write_length) if write_messages else []
    read_messages = system_prefix + visible_non_system + [dict(messages[target_index])]
    read = tokenize_messages_for_sft(
        tokenizer,
        read_messages,
        max_length=max_length,
        assistant_loss_mode="final_assistant_only",
    )
    fallback_messages = system_prefix + visible_non_system + [
        {
            **dict(messages[target_index]),
            "content": fallback_answer,
        }
    ]
    fallback = tokenize_messages_for_sft(
        tokenizer,
        fallback_messages,
        max_length=max_length,
        assistant_loss_mode="final_assistant_only",
    )
    input_ids = read["input_ids"]
    labels = read["labels"]
    if len(input_ids) < 2:
        raise ValueError("episode read example is too short for next-token loss")
    if not any(label != IGNORE_INDEX for label in labels[1:]):
        raise ValueError("episode final assistant label span is empty after shifting/truncation")
    pad_token_id = int(getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None) or 0)
    padded_ids = input_ids + [pad_token_id] * (max_length - len(input_ids))
    padded_labels = labels + [IGNORE_INDEX] * (max_length - len(labels))
    fallback_ids = fallback["input_ids"]
    fallback_labels = fallback["labels"]
    padded_fallback_ids = fallback_ids + [pad_token_id] * (max_length - len(fallback_ids))
    padded_fallback_labels = fallback_labels + [IGNORE_INDEX] * (max_length - len(fallback_labels))
    return Batch(
        input_ids=mx.array([padded_ids], dtype=mx.int32),
        labels=mx.array([padded_labels], dtype=mx.int32),
        write_input_ids=mx.array([write_input_ids], dtype=mx.int32) if write_input_ids else None,
        fallback_input_ids=mx.array([padded_fallback_ids], dtype=mx.int32),
        fallback_labels=mx.array([padded_fallback_labels], dtype=mx.int32),
        shape_signature={
            "mode": "episode",
            "write_tokens": len(write_input_ids),
            "read_tokens": len(input_ids),
            "fallback_read_tokens": len(fallback_ids),
            "supervised_tokens": sum(1 for label in labels if label != IGNORE_INDEX),
            "fallback_supervised_tokens": sum(1 for label in fallback_labels if label != IGNORE_INDEX),
            "episode_recent_messages": episode_recent_messages,
            "max_write_length": max_write_length,
        },
    )


def tokenize_messages_for_sft(
    tokenizer: Any,
    messages: list[dict[str, str]],
    max_length: int,
    *,
    assistant_loss_mode: str,
) -> dict[str, list[int]]:
    supervised_assistant_indices = set(select_supervised_assistant_indices(messages, assistant_loss_mode))
    input_ids: list[int] = []
    labels: list[int] = []
    previous_ids: list[int] = []
    for index in range(len(messages)):
        if index == 0 and messages[index].get("role") == "system":
            continue
        current_ids = tokenize_chat_messages(tokenizer, messages[: index + 1])
        prefix_len, delta_ids = chat_template_delta(
            previous_ids,
            current_ids,
            error_message="Chat template tokenization is not prefix-stable; cannot build assistant-span labels safely.",
        )
        if prefix_len < len(input_ids):
            del input_ids[prefix_len:]
            del labels[prefix_len:]
        input_ids.extend(delta_ids)
        labels.extend(delta_ids if index in supervised_assistant_indices else [IGNORE_INDEX] * len(delta_ids))
        previous_ids = current_ids
    input_ids, labels = truncate_sft_sequence(input_ids, labels, max_length)
    return {"input_ids": input_ids, "labels": labels}


def split_system_prefix(messages: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    system_prefix: list[dict[str, str]] = []
    index = 0
    while index < len(messages) and messages[index].get("role") == "system":
        system_prefix.append(messages[index])
        index += 1
    return system_prefix, messages[index:]


def truncate_token_sequence(input_ids: list[int], max_length: int) -> list[int]:
    if max_length <= 0:
        raise ValueError("max_length must be > 0")
    if len(input_ids) <= max_length:
        return input_ids
    return input_ids[-max_length:]


def select_supervised_assistant_indices(messages: list[dict[str, str]], assistant_loss_mode: str) -> list[int]:
    assistant_indices = [index for index, message in enumerate(messages) if message["role"] == "assistant"]
    if not assistant_indices:
        raise ValueError("No assistant turns found for supervision")
    if assistant_loss_mode == "all_assistant_turns":
        return assistant_indices
    if assistant_loss_mode == "final_assistant_only":
        return [assistant_indices[-1]]
    raise ValueError(f"Unsupported assistant_loss_mode: {assistant_loss_mode}")


def tokenize_chat_messages(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    if hasattr(tokenizer, "apply_chat_template"):
        tokenized = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        if hasattr(tokenized, "input_ids"):
            tokenized = tokenized.input_ids
        if hasattr(tokenized, "squeeze"):
            tokenized = tokenized.squeeze(0)
        if hasattr(tokenized, "tolist"):
            tokenized = tokenized.tolist()
        return list(tokenized)
    text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    return list(tokenizer.encode(text))


def chat_template_delta(previous_ids: list[int], current_ids: list[int], *, error_message: str) -> tuple[int, list[int]]:
    prefix_len = longest_common_prefix_length(previous_ids, current_ids)
    rollback_tokens = len(previous_ids) - prefix_len
    if rollback_tokens > MAX_CHAT_TEMPLATE_SUFFIX_ROLLBACK_TOKENS:
        raise ValueError(error_message)
    return prefix_len, current_ids[prefix_len:]


def longest_common_prefix_length(left: list[int], right: list[int]) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def truncate_sft_sequence(input_ids: list[int], labels: list[int], max_length: int) -> tuple[list[int], list[int]]:
    if max_length <= 0:
        raise ValueError("max_length must be > 0")
    if len(input_ids) <= max_length:
        return input_ids, labels
    start = len(input_ids) - max_length
    return input_ids[start:], labels[start:]


def freeze_base_model_contract(model: Any) -> None:
    if hasattr(model, "freeze"):
        model.freeze()
    for layer_index, attention in iter_attention_layers(model):
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            projection = getattr(attention, name, None)
            if projection is None:
                raise RuntimeError(f"layer {layer_index} missing {name}")
            if "QuantizedLinear" not in type(projection).__name__:
                raise RuntimeError(f"layer {layer_index} {name} is {type(projection).__name__}, expected QuantizedLinear")


def iter_attention_layers(model: Any) -> Iterable[tuple[int, Any]]:
    inner = getattr(model, "model", model)
    for layer_index, layer in enumerate(getattr(inner, "layers")):
        attention = getattr(layer, "self_attn", None)
        if attention is not None and all(hasattr(attention, name) for name in ("q_proj", "k_proj", "v_proj", "o_proj")):
            yield layer_index, attention


def init_adapter_tree(
    model: Any,
    *,
    target_layers: tuple[int, ...],
    target_linear_layers: tuple[int, ...],
    rank: int,
    beta_bias_init: float,
    nn: Any,
    mx: Any,
) -> Any:
    layers: dict[str, Any] = {}
    output_layers: dict[str, Any] = {}
    for layer_index, attention in iter_attention_layers(model):
        if layer_index not in target_layers:
            continue
        hidden = infer_hidden_size(attention)
        q_out = infer_effective_q_out(attention)
        scale = 1.0 / math.sqrt(hidden)
        layers[str(layer_index)] = {
            "memory_q_proj": mx.random.normal((rank, hidden)) * scale,
            "memory_k_proj": mx.random.normal((rank, hidden)) * scale,
            "memory_v_proj": mx.random.normal((rank, hidden)) * scale,
            "delta_q_proj": mx.zeros((q_out, rank)),
            "delta_o_proj": mx.zeros((hidden, rank)),
            "beta_proj": mx.zeros((rank, hidden)),
            "beta_bias": mx.full((rank,), beta_bias_init),
        }
    missing = sorted(set(target_layers) - {int(k) for k in layers})
    if missing:
        raise RuntimeError(f"target layers were not full-attention layers: {missing}")
    for layer_index, linear_attn in iter_linear_layers(model):
        if layer_index not in target_linear_layers:
            continue
        hidden = infer_linear_hidden_size(linear_attn)
        scale = 1.0 / math.sqrt(hidden)
        output_layers[str(layer_index)] = {
            "memory_q_proj": mx.random.normal((rank, hidden)) * scale,
            "memory_k_proj": mx.random.normal((rank, hidden)) * scale,
            "memory_v_proj": mx.random.normal((rank, hidden)) * scale,
            "delta_o_proj": mx.zeros((hidden, rank)),
            "beta_proj": mx.zeros((rank, hidden)),
            "beta_bias": mx.full((rank,), beta_bias_init),
        }
    missing_linear = sorted(set(target_linear_layers) - {int(k) for k in output_layers})
    if missing_linear:
        raise RuntimeError(f"target linear layers were not GatedDeltaNet layers: {missing_linear}")

    class DeltaMemAdapterModule(nn.Module):
        def __init__(self, layer_tree: dict[str, Any], output_layer_tree: dict[str, Any]) -> None:
            super().__init__()
            self.layers = layer_tree
            self.output_layers = output_layer_tree

    return DeltaMemAdapterModule(layers, output_layers)


def iter_linear_layers(model: Any) -> Iterable[tuple[int, Any]]:
    inner = getattr(model, "model", model)
    for layer_index, layer in enumerate(getattr(inner, "layers")):
        if getattr(layer, "is_linear", False) and hasattr(layer, "linear_attn"):
            yield layer_index, layer.linear_attn


def infer_hidden_size(attention: Any) -> int:
    o_out = _linear_output_dims(getattr(attention, "o_proj", None))
    if o_out is None:
        raise RuntimeError("Could not infer attention hidden size")
    return int(o_out)


def infer_effective_q_out(attention: Any) -> int:
    q_out = _linear_output_dims(getattr(attention, "q_proj", None))
    hidden = infer_hidden_size(attention)
    if q_out is None:
        raise RuntimeError("Could not infer attention q projection width")
    if int(q_out) == hidden * 2 or "qwen3next" in type(attention).__name__.lower():
        return int(q_out) // 2
    return int(q_out)


def infer_linear_hidden_size(linear_attn: Any) -> int:
    hidden_size = getattr(linear_attn, "hidden_size", None)
    if hidden_size is None:
        raise RuntimeError(f"Could not infer GatedDeltaNet hidden size for {type(linear_attn).__name__}")
    return int(hidden_size)


def assert_optimizer_only_adapter_contract(model: Any, adapter: Any) -> None:
    if not adapter.layers and not adapter.output_layers:
        raise RuntimeError("no adapter tensors were initialized")
    for name, tensor in flat_tree(adapter.trainable_parameters()).items():
        if not any(name.endswith(key) for key in ADAPTER_KEYS + OUTPUT_ADAPTER_KEYS):
            raise RuntimeError(f"unexpected optimizer tensor in adapter tree: {name}")
        if not hasattr(tensor, "shape"):
            raise RuntimeError(f"adapter leaf is not a tensor: {name}")


def install_trainable_delta_attention(model: Any, adapter: Any, *, alpha: float, rank: int, max_write_length: int) -> None:
    for layer_index, params in adapter.layers.items():
        idx = int(layer_index)
        attention = get_attention(model, idx)
        if isinstance(attention, TrainableDeltaAttention):
            raise RuntimeError(f"layer {idx} already has an installed trainable delta attention wrapper")
        set_attention(
            model,
            idx,
            TrainableDeltaAttention(attention, params, alpha=alpha, rank=rank, max_write_length=max_write_length),
        )
    for layer_index, params in adapter.output_layers.items():
        idx = int(layer_index)
        linear_attn = get_linear_attention(model, idx)
        if isinstance(linear_attn, TrainableDeltaGatedDeltaNetOutput):
            raise RuntimeError(f"layer {idx} already has an installed trainable GatedDeltaNet output wrapper")
        set_linear_attention(
            model,
            idx,
            TrainableDeltaGatedDeltaNetOutput(linear_attn, params, alpha=alpha, rank=rank, max_write_length=max_write_length),
        )


def reset_trainable_delta_states(model: Any) -> None:
    for _, attention in iter_attention_layers(model):
        if hasattr(attention, "reset_state"):
            attention.reset_state()
    for _, attention in iter_linear_layers(model):
        if hasattr(attention, "reset_state"):
            attention.reset_state()


def set_trainable_delta_write_enabled(model: Any, enabled: bool) -> None:
    for _, attention in iter_attention_layers(model):
        if hasattr(attention, "set_write_enabled"):
            attention.set_write_enabled(enabled)
    for _, attention in iter_linear_layers(model):
        if hasattr(attention, "set_write_enabled"):
            attention.set_write_enabled(enabled)


class TrainableDeltaAttention:
    def __init__(self, base: Any, params: dict[str, Any], *, alpha: float, rank: int, max_write_length: int) -> None:
        self.base = base
        self.params = params
        self.config = MlxDeltaAttentionConfig(rank=rank, alpha=alpha)
        self.max_write_length = max_write_length
        weights = MlxDeltaAttentionWeights(
            memory_q_proj=params["memory_q_proj"],
            memory_k_proj=params["memory_k_proj"],
            memory_v_proj=params["memory_v_proj"],
            delta_q_proj=params["delta_q_proj"],
            delta_o_proj=params["delta_o_proj"],
            beta_proj=params["beta_proj"],
            beta_bias=params["beta_bias"],
        )
        shape = infer_mlx_attention_shape(base, weights)
        self.n_heads = shape.n_heads
        self.n_kv_heads = shape.n_kv_heads
        self.head_dim = shape.head_dim
        self.query_out_dims = shape.query_out_dims
        self.is_query_gated = shape.is_query_gated
        self.scale = base.scale
        self.q_proj = base.q_proj
        self.k_proj = base.k_proj
        self.v_proj = base.v_proj
        self.o_proj = base.o_proj
        self.rope = base.rope
        self.q_norm = getattr(base, "q_norm", None)
        self.k_norm = getattr(base, "k_norm", None)
        self.delta_state: Any | None = None
        self.write_enabled = True

    def reset_state(self) -> None:
        self.delta_state = None

    def set_write_enabled(self, enabled: bool) -> None:
        self.write_enabled = enabled

    def __call__(self, x: Any, mask: Any | None = None, cache: Any | None = None) -> Any:
        import mlx.core as mx
        from mlx_lm.models.qwen2 import scaled_dot_product_attention

        batch_size, seq_len, _ = x.shape
        reads = self.memory_reads(x, mx)
        queries, query_gate = self.query_projection_with_optional_gate(x, mx)
        queries = queries + linear(reads, self.params["delta_q_proj"], mx).astype(queries.dtype) * self.config.delta_scaling
        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = queries.reshape(batch_size, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = keys.reshape(batch_size, seq_len, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(batch_size, seq_len, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        if self.q_norm is not None:
            queries = self.q_norm(queries)
        if self.k_norm is not None:
            keys = self.k_norm(keys)
        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(queries, keys, values, cache=cache, scale=self.scale, mask=mask)
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        if self.is_query_gated:
            if query_gate is None:
                raise RuntimeError("Qwen3.5 query-gated attention did not capture a query gate")
            output = output * mx.sigmoid(query_gate).astype(output.dtype)
        output = self.o_proj(output)
        output = output + linear(reads, self.params["delta_o_proj"], mx).astype(output.dtype) * self.config.delta_scaling
        return output

    def query_projection_with_optional_gate(self, x: Any, mx: Any) -> tuple[Any, Any | None]:
        projected = self.q_proj(x)
        if not self.is_query_gated:
            return projected, None
        batch_size, seq_len, _ = projected.shape
        projected = projected.reshape(batch_size, seq_len, self.n_heads, self.head_dim * 2)
        query, gate = mx.split(projected, 2, axis=-1)
        return (
            query.reshape(batch_size, seq_len, self.query_out_dims),
            gate.reshape(batch_size, seq_len, self.query_out_dims),
        )

    def memory_reads(self, x: Any, mx: Any) -> Any:
        memory_q = _normalize_memory_projection(linear(x, self.params["memory_q_proj"], mx))
        memory_k = _normalize_memory_projection(linear(x, self.params["memory_k_proj"], mx))
        memory_v = linear(x, self.params["memory_v_proj"], mx)
        beta = mx.sigmoid(linear(x, self.params["beta_proj"], mx) + self.params["beta_bias"])
        lam = 1.0 - beta
        state = self.ensure_state(int(x.shape[0]), memory_q.dtype, mx)
        write_len = min(int(self.max_write_length), int(x.shape[1]))
        if not self.write_enabled or write_len <= 0:
            return mx.matmul(state, mx.expand_dims(memory_q, -1)).squeeze(-1)
        if write_len == int(x.shape[1]):
            state, reads = delta_affine_scan(state, memory_q, memory_k, memory_v, lam, beta, beta)
            self.delta_state = state
            return reads

        state, prefix_reads = delta_affine_scan(
            state,
            memory_q[:, :write_len, :],
            memory_k[:, :write_len, :],
            memory_v[:, :write_len, :],
            lam[:, :write_len, :],
            beta[:, :write_len, :],
            beta[:, :write_len, :],
        )
        self.delta_state = state
        suffix_reads = mx.matmul(state, mx.expand_dims(memory_q[:, write_len:, :], -1)).squeeze(-1)
        return mx.concatenate([prefix_reads, suffix_reads], axis=1)

    def ensure_state(self, batch_size: int, dtype: Any, mx: Any) -> Any:
        shape = (batch_size, self.config.rank, self.config.rank)
        if self.delta_state is None or tuple(self.delta_state.shape) != shape:
            self.delta_state = mx.zeros(shape, dtype=dtype)
        return self.delta_state


class TrainableDeltaGatedDeltaNetOutput:
    def __init__(self, base: Any, params: dict[str, Any], *, alpha: float, rank: int, max_write_length: int) -> None:
        self.base = base
        self.params = params
        self.config = MlxDeltaAttentionConfig(rank=rank, alpha=alpha)
        self.max_write_length = max_write_length
        self.hidden_size = infer_linear_hidden_size(base)
        self.delta_state: Any | None = None
        self.write_enabled = True

    def reset_state(self) -> None:
        self.delta_state = None

    def set_write_enabled(self, enabled: bool) -> None:
        self.write_enabled = enabled

    def __call__(self, inputs: Any, mask: Any | None = None, cache: Any | None = None) -> Any:
        import mlx.core as mx

        base_output = self.base(inputs, mask=mask, cache=cache)
        reads = self.memory_reads(inputs, mx)
        output = base_output + linear(reads, self.params["delta_o_proj"], mx).astype(base_output.dtype) * self.config.delta_scaling
        return output

    def memory_reads(self, x: Any, mx: Any) -> Any:
        memory_q = _normalize_memory_projection(linear(x, self.params["memory_q_proj"], mx))
        memory_k = _normalize_memory_projection(linear(x, self.params["memory_k_proj"], mx))
        memory_v = linear(x, self.params["memory_v_proj"], mx)
        beta = mx.sigmoid(linear(x, self.params["beta_proj"], mx) + self.params["beta_bias"])
        lam = 1.0 - beta
        state = self.ensure_state(int(x.shape[0]), memory_q.dtype, mx)
        write_len = min(int(self.max_write_length), int(x.shape[1]))
        if not self.write_enabled or write_len <= 0:
            return mx.matmul(state, mx.expand_dims(memory_q, -1)).squeeze(-1)
        if write_len == int(x.shape[1]):
            state, reads = delta_affine_scan(state, memory_q, memory_k, memory_v, lam, beta, beta)
            self.delta_state = state
            return reads
        state, prefix_reads = delta_affine_scan(
            state,
            memory_q[:, :write_len, :],
            memory_k[:, :write_len, :],
            memory_v[:, :write_len, :],
            lam[:, :write_len, :],
            beta[:, :write_len, :],
            beta[:, :write_len, :],
        )
        self.delta_state = state
        suffix_reads = mx.matmul(state, mx.expand_dims(memory_q[:, write_len:, :], -1)).squeeze(-1)
        return mx.concatenate([prefix_reads, suffix_reads], axis=1)

    def ensure_state(self, batch_size: int, dtype: Any, mx: Any) -> Any:
        shape = (batch_size, self.config.rank, self.config.rank)
        if self.delta_state is None or tuple(self.delta_state.shape) != shape:
            self.delta_state = mx.zeros(shape, dtype=dtype)
        return self.delta_state


def get_attention(model: Any, layer_index: int) -> Any:
    return getattr(getattr(model, "model", model).layers[layer_index], "self_attn")


def set_attention(model: Any, layer_index: int, attention: Any) -> None:
    getattr(model, "model", model).layers[layer_index].self_attn = attention


def get_linear_attention(model: Any, layer_index: int) -> Any:
    layer = getattr(model, "model", model).layers[layer_index]
    if not getattr(layer, "is_linear", False) or not hasattr(layer, "linear_attn"):
        raise RuntimeError(f"layer {layer_index} is not a GatedDeltaNet layer")
    return layer.linear_attn


def set_linear_attention(model: Any, layer_index: int, attention: Any) -> None:
    layer = getattr(model, "model", model).layers[layer_index]
    if not getattr(layer, "is_linear", False) or not hasattr(layer, "linear_attn"):
        raise RuntimeError(f"layer {layer_index} is not a GatedDeltaNet layer")
    layer.linear_attn = attention


def linear(x: Any, weight: Any, mx: Any) -> Any:
    return mx.matmul(x, weight.T)


def final_assistant_cross_entropy(logits: Any, labels: Any, nn: Any, mx: Any) -> Any:
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    valid = shift_labels != IGNORE_INDEX
    safe_labels = mx.where(valid, shift_labels, mx.zeros_like(shift_labels))
    token_loss = nn.losses.cross_entropy(shift_logits, safe_labels, reduction="none")
    denom = mx.maximum(mx.sum(valid.astype(token_loss.dtype)), mx.array(1.0, dtype=token_loss.dtype))
    return mx.sum(mx.where(valid, token_loss, mx.zeros_like(token_loss))) / denom


def focused_answer_cross_entropy(logits: Any, labels: Any, nn: Any, mx: Any, *, answer_focus_tokens: int) -> Any:
    if answer_focus_tokens <= 0:
        raise ValueError("--answer-focus-tokens must be > 0")
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    valid = shift_labels != IGNORE_INDEX
    valid_rank = mx.cumsum(valid.astype(mx.int32), axis=1)
    focus_mask = valid & (valid_rank <= answer_focus_tokens)
    safe_labels = mx.where(focus_mask, shift_labels, mx.zeros_like(shift_labels))
    token_loss = nn.losses.cross_entropy(shift_logits, safe_labels, reduction="none")
    denom = mx.maximum(mx.sum(focus_mask.astype(token_loss.dtype)), mx.array(1.0, dtype=token_loss.dtype))
    return mx.sum(mx.where(focus_mask, token_loss, mx.zeros_like(token_loss))) / denom


def combined_training_loss(
    logits: Any,
    labels: Any,
    nn: Any,
    mx: Any,
    *,
    answer_focus_weight: float,
    answer_focus_tokens: int,
) -> Any:
    base_loss = final_assistant_cross_entropy(logits, labels, nn, mx)
    if answer_focus_weight <= 0.0:
        return base_loss
    focus_loss = focused_answer_cross_entropy(
        logits,
        labels,
        nn,
        mx,
        answer_focus_tokens=answer_focus_tokens,
    )
    return base_loss + answer_focus_weight * focus_loss


def memory_margin_objective(gap: Any, margin: float, mx: Any) -> Any:
    if margin <= 0:
        raise ValueError("--memory-margin must be > 0")
    scaled_gap = (margin - gap) / margin
    return mx.logaddexp(scaled_gap, mx.array(0.0, dtype=scaled_gap.dtype))


def pairwise_margin_objective(gap: Any, margin: float, mx: Any) -> Any:
    if margin <= 0:
        raise ValueError("pairwise margin must be > 0")
    scaled_gap = (margin - gap) / margin
    return mx.logaddexp(scaled_gap, mx.array(0.0, dtype=scaled_gap.dtype))


def normalize_answer_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def require_finite(name: str, tensor: Any, mx: Any) -> None:
    if not bool(mx.all(mx.isfinite(tensor)).item()):
        raise RuntimeError(f"non-finite {name}")


def require_finite_tree(name: str, tree: dict[str, Any], mx: Any) -> None:
    for key, tensor in flat_tree(tree).items():
        require_finite(f"{name}.{key}", tensor, mx)


def flat_tree(tree: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(tree, dict):
        result = {}
        for key, value in tree.items():
            result.update(flat_tree(value, f"{prefix}.{key}" if prefix else str(key)))
        return result
    return {prefix: tree}


def tree_copy(tree: dict[str, Any], mx: Any) -> dict[str, Any]:
    return {key: tree_copy(value, mx) if isinstance(value, dict) else mx.array(value) for key, value in tree.items()}


def adapter_tree_changed(before: dict[str, Any], after: dict[str, Any], mx: Any) -> bool:
    after_flat = flat_tree(after)
    for key, old in flat_tree(before).items():
        if bool(mx.any(mx.abs(after_flat[key] - old) > 0).item()):
            return True
    return False


def export_adapter_npz(path: Path, adapter: Any, mx: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for layer_index, params in sorted(adapter.layers.items(), key=lambda item: int(item[0])):
        prefix = f"model.layers.{layer_index}.self_attn"
        for name in ADAPTER_KEYS:
            arrays[f"{prefix}.{name}"] = params[name]
    for layer_index, params in sorted(adapter.output_layers.items(), key=lambda item: int(item[0])):
        prefix = f"model.layers.{layer_index}.linear_attn"
        for name in OUTPUT_ADAPTER_KEYS:
            arrays[f"{prefix}.{name}"] = params[name]
    mx.savez(str(path), **arrays)


def build_training_summary(
    *,
    args: argparse.Namespace,
    target_layers: tuple[int, ...],
    target_linear_layers: tuple[int, ...],
    compiled: bool,
    loss_history: list[float],
    step_times: list[dict[str, Any]],
    changed: bool | None,
    train_elapsed: float,
    report: dict[str, Any],
    checkpoint_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    summary = {
        "model": args.model,
        "train_file": str(args.train_file),
        "target_layers": list(target_layers),
        "target_linear_layers": list(target_linear_layers),
        "rank": args.rank,
        "alpha": args.alpha,
        "training_mode": args.training_mode,
        "episode_recent_messages": args.episode_recent_messages,
        "dataset_limit": args.dataset_limit,
        "shuffle_examples": args.shuffle_examples,
        "grad_checkpoint": args.grad_checkpoint,
        "clear_cache_every": args.clear_cache_every,
        "memory_contrast_weight": args.memory_contrast_weight,
        "memory_contrast_every": args.memory_contrast_every,
        "memory_margin": args.memory_margin,
        "answer_focus_weight": args.answer_focus_weight,
        "answer_focus_tokens": args.answer_focus_tokens,
        "fallback_answer": args.fallback_answer,
        "exclude_fallback_answers": args.exclude_fallback_answers,
        "fallback_contrast_weight": args.fallback_contrast_weight,
        "fallback_contrast_every": args.fallback_contrast_every,
        "fallback_margin": args.fallback_margin,
        "fallback_gap_clip": args.fallback_gap_clip,
        "max_length": args.max_length,
        "max_write_length": args.max_write_length,
        "max_steps": args.max_steps,
        "current_step": len(loss_history),
        "compile_step": compiled,
        "log_every": args.log_every,
        "checkpoint_every": args.checkpoint_every,
        "learning_rate": args.learning_rate,
        "loss_history": loss_history,
        "step_times": step_times,
        "adapter_changed": changed,
        "elapsed_s": train_elapsed,
        "steps_per_second": (len(loss_history) / train_elapsed) if train_elapsed > 0 else None,
        "timing_seconds": report["timing_seconds"],
        "memory": report["memory"],
        "shape_signatures": report["shape_signatures"],
    }
    if checkpoint_records is not None:
        summary["checkpoints"] = checkpoint_records
    return summary


def write_training_artifacts(
    *,
    output_dir: Path,
    adapter: Any,
    args: argparse.Namespace,
    target_layers: tuple[int, ...],
    target_linear_layers: tuple[int, ...],
    summary: dict[str, Any],
    report: dict[str, Any],
    mx: Any,
    validate: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    export_adapter_npz(output_dir / "delta_mem_adapter_mlx.npz", adapter, mx)
    write_config(output_dir / "delta_mem_config.json", args, target_layers, target_linear_layers)
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    benchmark = {
        **report,
        "artifact_dir": str(output_dir),
        "written_at": utc_now_iso(),
    }
    (output_dir / "benchmark.json").write_text(json.dumps(benchmark, indent=2, sort_keys=True) + "\n")
    if validate:
        validate_export_files(output_dir, adapter, target_layers, target_linear_layers, args.rank, mx)
        validate_with_sidecar_loader(output_dir)


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def write_config(
    path: Path,
    args: argparse.Namespace,
    target_layers: tuple[int, ...],
    target_linear_layers: tuple[int, ...],
) -> None:
    path.write_text(
        json.dumps(
            {
                "base_model": args.model,
                "rank": args.rank,
                "alpha": args.alpha,
                "target_layers": list(target_layers),
                "target_linear_layers": list(target_linear_layers),
                "delta_heads": ["q", "o"],
                "answer_focus_weight": args.answer_focus_weight,
                "answer_focus_tokens": args.answer_focus_tokens,
                "fallback_answer": args.fallback_answer,
                "exclude_fallback_answers": args.exclude_fallback_answers,
                "fallback_contrast_weight": args.fallback_contrast_weight,
                "fallback_contrast_every": args.fallback_contrast_every,
                "fallback_margin": args.fallback_margin,
                "fallback_gap_clip": args.fallback_gap_clip,
                "num_state_heads": 1,
                "num_memory_partitions": 1,
                "num_global_memory_partitions": 0,
                "normalize_qk": True,
                "couple_lambda": True,
                "state_update_mode": "standard",
                "memory_readout_mode": "delta",
                "memory_write_source": "learned_hidden",
                "memory_write_granularity": "token",
                "delta_o_rmsnorm": False,
                "trainable_delta_scale": False,
                "format": "delta_mem_adapter_mlx.npz",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def validate_export_files(
    output_dir: Path,
    adapter: Any,
    target_layers: tuple[int, ...],
    target_linear_layers: tuple[int, ...],
    rank: int,
    mx: Any,
) -> None:
    for filename in ("delta_mem_adapter_mlx.npz", "delta_mem_config.json", "training_summary.json"):
        if not (output_dir / filename).exists():
            raise RuntimeError(f"missing export file: {filename}")
    arrays = dict(mx.load(str(output_dir / "delta_mem_adapter_mlx.npz")))
    expected = {f"model.layers.{layer}.self_attn.{name}" for layer in target_layers for name in ADAPTER_KEYS}
    expected.update(
        f"model.layers.{layer}.linear_attn.{name}"
        for layer in target_linear_layers
        for name in OUTPUT_ADAPTER_KEYS
    )
    if set(arrays) != expected:
        raise RuntimeError(f"export key mismatch: missing={sorted(expected - set(arrays))}, extra={sorted(set(arrays) - expected)}")
    for key, tensor in arrays.items():
        require_finite(f"export.{key}", tensor, mx)
        if key.endswith("beta_bias") and tuple(tensor.shape) != (rank,):
            raise RuntimeError(f"{key} shape {tensor.shape} != {(rank,)}")
    require_finite_tree("export_source_adapter", adapter.trainable_parameters(), mx)


def validate_with_sidecar_loader(output_dir: Path) -> None:
    from delta_mem_sidecar.mlx_delta_adapter import load_mlx_delta_adapter

    adapter = load_mlx_delta_adapter(output_dir)
    if not adapter.weights_by_layer and not adapter.output_weights_by_layer:
        raise RuntimeError("sidecar loader returned no adapter layers")


def memory_snapshot(mx: Any) -> dict[str, Any]:
    snapshot = {"rss_max_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2)}
    metal = getattr(mx, "metal", None)
    for name in ("get_active_memory", "get_peak_memory", "get_cache_memory"):
        fn = getattr(mx, name, None) or getattr(metal, name, None)
        if callable(fn):
            snapshot[name.removeprefix("get_") + "_mb"] = round(fn() / 1024**2, 2)
    return snapshot


if __name__ == "__main__":
    main()
