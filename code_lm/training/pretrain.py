from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
import random
import shutil

import numpy as np
import torch

from code_lm.training.checkpoint import load_checkpoint, save_checkpoint
from code_lm.data.pretrain import ShuffledTokenBlocks, get_batch
from code_lm.training.schedule import cosine_learning_rate_schedule
from code_lm.training.common import autocast_context, evaluate
from torch.nn import functional as F
from code_lm.model import TransformerLM, model_args_from_config, inference_weights


@dataclass(frozen=True)
class TrainingConfig:
    train_data: Path
    val_data: Path
    checkpoint_path: Path
    resume_from: Path | None
    data_dtype: str
    device: str
    seed: int
    vocab_size: int
    context_length: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    rope_theta: float
    batch_size: int
    total_steps: int
    max_learning_rate: float
    min_learning_rate: float
    warmup_steps: int
    weight_decay: float
    max_grad_norm: float | None
    log_every: int
    eval_every: int
    eval_batches: int
    checkpoint_every: int
    precision: str = "auto"
    grad_accum_steps: int = 1
    compile: bool = False
    data_order: str = "random"
    data_seed: int = 2026
    validation_seed: int = 1234
    timing_warmup_steps: int = 0
    metrics_path: Path | None = None
    architecture: str = "standard"
    prelude_layers: int = 0
    recurrent_layers: int = 0
    coda_layers: int = 0
    loop_count: int = 1
    init_from: Path | None = None
    loop_max_count: int = 0
    loop_lora_rank: int = 0
    loop_lora_alpha: float = 8.0
    compile_loss: bool = False


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the APS360 Python language model on token arrays.")
    parser.add_argument("--config", type=Path, help="JSON recipe; explicit CLI flags override its values")
    parser.add_argument("--output-dir", type=Path, help="Directory for checkpoint, metrics, config, and final weights")
    parser.add_argument("--tokenizer-dir", type=Path, help="Optional directory with vocab.json and merges.txt to export")
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--train-data", type=Path)
    parser.add_argument("--val-data", type=Path)
    parser.add_argument("--checkpoint-path", type=Path, default=Path("checkpoints/latest.pt"))
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--init-from", type=Path, help="Continue pretrained weights with a fresh optimizer/schedule")
    parser.add_argument("--data-dtype", choices=("uint16", "uint32", "int32", "int64"), default="uint16")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    parser.add_argument("--architecture", choices=("standard", "loop"), default="standard")
    parser.add_argument("--prelude-layers", type=int, default=0)
    parser.add_argument("--recurrent-layers", type=int, default=0)
    parser.add_argument("--coda-layers", type=int, default=0)
    parser.add_argument("--loop-count", type=int, default=1)
    parser.add_argument('--loop-max-count', type=int, default=0)
    parser.add_argument('--loop-lora-rank', type=int, default=0)
    parser.add_argument('--loop-lora-alpha', type=float, default=8.)
    parser.add_argument('--compile-loss', action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--precision", choices=("auto", "fp32", "bf16"), default="auto")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--data-order", choices=("random", "shuffled_blocks"), default="random")
    parser.add_argument("--data-seed", type=int, default=2026)
    parser.add_argument("--validation-seed", type=int, default=1234)
    parser.add_argument("--timing-warmup-steps", type=int, default=0)
    parser.add_argument("--metrics-path", type=Path)
    parser.add_argument("--total-steps", type=int, default=5_000)
    parser.add_argument("--max-learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=500)

    # Read only the config flag first; all remaining flags are parsed once with recipe defaults.
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--config", type=Path)
    config_path = probe.parse_known_args(argv)[0].config
    if config_path is not None:
        recipe = json.loads(config_path.read_text())
        if not isinstance(recipe, dict):
            parser.error("Config must be a JSON object")
        actions = {a.dest: a for a in parser._actions if a.dest not in {"help", "config"}}
        unknown = recipe.keys() - actions.keys()
        if unknown:
            parser.error(f"Unknown config keys: {sorted(unknown)}")
        for key, value in recipe.items():
            action = actions[key]
            if value is not None:
                if isinstance(action, argparse.BooleanOptionalAction) and not isinstance(value, bool):
                    parser.error(f"{key} must be a JSON boolean")
                if action.type is not None:
                    value = action.type(value)
                if action.choices is not None and value not in action.choices:
                    parser.error(f"Invalid {key}: {value}")
            recipe[key] = value
        parser.set_defaults(**recipe)
    args = parser.parse_args(argv)
    if args.train_data is None or args.val_data is None:
        parser.error("--train-data and --val-data are required (CLI or config)")
    if args.num_threads <= 0:
        parser.error("--num-threads must be positive")
    if args.output_dir is not None:
        args.checkpoint_path = args.output_dir / "latest.pt"
        args.metrics_path = args.output_dir / "metrics.jsonl"
    return args


def load_token_data(path: Path, dtype: str) -> np.memmap:
    """Memory-map a raw one-dimensional token-ID file."""
    return np.memmap(path, dtype=np.dtype(dtype), mode="r")


def train(config: TrainingConfig, on_checkpoint: Callable[[], None] | None = None) -> dict:
    wall_start = time.perf_counter()
    if not 0 <= config.warmup_steps < config.total_steps:
        raise ValueError("warmup_steps must be smaller than total_steps")
    for name in ("batch_size", "context_length", "grad_accum_steps", "log_every", "eval_every", "eval_batches", "checkpoint_every"):
        if getattr(config, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if config.data_order not in ("random", "shuffled_blocks"):
        raise ValueError("Unknown data order")
    if not 0 <= config.timing_warmup_steps < config.total_steps:
        raise ValueError("timing_warmup_steps must be in [0, total_steps)")
    if config.compile_loss and not config.compile:
        raise ValueError('compile_loss requires compiled NTP training')

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    device = torch.device(config.device)
    use_bf16 = config.precision == "bf16" or (
        config.precision == "auto" and device.type == "cuda" and torch.cuda.is_bf16_supported()
    )
    if use_bf16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("This CUDA device does not support BF16; use --precision fp32")
    amp_dtype = torch.bfloat16 if use_bf16 else None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    train_data = load_token_data(config.train_data, config.data_dtype)
    val_data = load_token_data(config.val_data, config.data_dtype)
    blocks = ShuffledTokenBlocks(train_data, config.context_length, config.data_seed) if config.data_order == "shuffled_blocks" else None
    if blocks is not None and config.total_steps * config.batch_size * config.grad_accum_steps > len(blocks.order):
        raise ValueError("Training would repeat data; reduce total_steps")

    model_args = model_args_from_config(asdict(config))
    model = TransformerLM(**model_args, device=device)
    if config.init_from is not None and config.resume_from is None:
        saved = torch.load(config.init_from, map_location='cpu', weights_only=True)
        source = saved.get('model_args') or (saved.get('config') or {}).get('model_args') or saved.get('config')
        source_args = model_args_from_config(source or {})
        if source_args != model_args:
            raise ValueError('init_from architecture differs from continuation config')
        model.load_state_dict(saved['model'], strict=True)
        del saved
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in model.parameters() if p.ndim >= 2], "weight_decay": config.weight_decay},
            {"params": [p for p in model.parameters() if p.ndim < 2], "weight_decay": 0.0},
        ],
        lr=config.max_learning_rate,
        betas=(0.9, 0.95),
        fused=device.type == "cuda",
    )

    start_step = 0
    if config.resume_from is not None:
        saved = torch.load(config.resume_from, map_location="cpu", weights_only=True)
        saved_config = saved.get("config")
        if saved_config:
            if model_args_from_config(saved_config.get("model_args") or saved_config) != model_args:
                raise ValueError("Resume model architecture differs, including loop layout/count")
            # A continuation must retain its schedule, data, and effective batch.
            mutable = {"resume_from", "checkpoint_path", "device", "compile", "compile_loss", "log_every", "eval_every", "eval_batches", "checkpoint_every", "metrics_path", "timing_warmup_steps"}
            current = {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()}
            mismatches = [k for k, v in saved_config.items() if k not in mutable and current.get(k) != v]
            if mismatches:
                raise ValueError(f"Resume configuration differs: {', '.join(mismatches)}")
        elif config.architecture != "standard":
            raise ValueError("Cannot resume a loop model without saved architecture metadata")
        start_step = load_checkpoint(config.resume_from, model, optimizer)
    execution_model = torch.compile(model, dynamic=False) if config.compile and not config.compile_loss else model
    def loss_forward(inputs, targets, depth):
        logits = model(inputs, loop_count=depth)
        return F.cross_entropy(logits.float().flatten(0, 1), targets.flatten())
    # Python integer depths specialize to three fixed graphs; no tensor-driven loop.
    compiled_loss = torch.compile(loss_forward, dynamic=False) if config.compile_loss else None
    tokens_per_step = config.batch_size * config.context_length * config.grad_accum_steps
    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()}
    print(f"architecture={model.architecture} unique_layers={len(model.layers)} effective_depth={model.effective_depth} "
          f"parameters={sum(p.numel() for p in model.parameters()):,} precision={'bf16' if use_bf16 else 'fp32'} tokens/step={tokens_per_step:,}")

    def save(step: int) -> None:
        save_checkpoint(model, optimizer, step, config.checkpoint_path, config=metadata, tokens_seen=step * tokens_per_step)
        if on_checkpoint is not None:
            on_checkpoint()

    def emit(record: dict) -> None:
        if config.metrics_path is not None:
            config.metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with config.metrics_path.open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()

    model.train()
    synchronize()
    interval_start = time.perf_counter()
    interval_steps = 0
    interval_first_step = start_step
    training_seconds = stable_seconds = evaluation_seconds = checkpoint_seconds = 0.0
    stable_steps = 0
    first_step_seconds = None
    final_loss = final_train_loss = None
    for step in range(start_step, config.total_steps):
        depth = config.loop_count
        learning_rate = cosine_learning_rate_schedule(
            iteration=step,
            max_learning_rate=config.max_learning_rate,
            min_learning_rate=config.min_learning_rate,
            warmup_iters=config.warmup_steps,
            cosine_cycle_iters=config.total_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        optimizer.zero_grad(set_to_none=True)
        loss = torch.zeros((), device=device)
        ntp_total = torch.zeros((), device=device)
        batches = []
        for micro_step in range(config.grad_accum_steps):
            if blocks is None:
                inputs, targets = get_batch(train_data, config.batch_size, config.context_length, device)
            else:
                offset = (step * config.grad_accum_steps + micro_step) * config.batch_size
                inputs, targets = blocks.batch(offset, config.batch_size, device)
            batches.append((inputs, targets))
        for inputs, targets in batches:
            with autocast_context(device, amp_dtype):
                if compiled_loss is not None:
                    ntp = compiled_loss(inputs, targets, depth) / config.grad_accum_steps
                else:
                    logits = execution_model(inputs, loop_count=depth)
                    ntp = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten()) / config.grad_accum_steps
                micro_loss = ntp
                ntp_total += ntp.detach()
            micro_loss.backward()
            loss += micro_loss.detach()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.max_grad_norm if config.max_grad_norm is not None else float("inf"),
            error_if_nonfinite=True,
        )
        optimizer.step()
        completed_steps = step + 1
        interval_steps += 1

        do_eval = completed_steps % config.eval_every == 0 or completed_steps == config.total_steps
        do_save = completed_steps % config.checkpoint_every == 0 or completed_steps == config.total_steps
        boundary = (completed_steps % config.log_every == 0 or completed_steps in (1, config.timing_warmup_steps, config.total_steps) or do_eval or do_save)
        if boundary:
            synchronize()
            elapsed = time.perf_counter() - interval_start
            training_seconds += elapsed
            if completed_steps == 1:
                first_step_seconds = elapsed
            if interval_first_step >= config.timing_warmup_steps:
                stable_seconds += elapsed
                stable_steps += interval_steps
            throughput = interval_steps * tokens_per_step / elapsed
            memory = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
            final_train_loss = float(loss)
            if not math.isfinite(final_train_loss):
                raise RuntimeError("Non-finite training loss")
            print(
                f"step={completed_steps} "
                f"train_loss={float(loss):.4f} "
                f"lr={learning_rate:.3e} grad_norm={float(grad_norm):.3f} "
                f"tokens/s={throughput:,.0f} peak_cuda_gib={memory:.2f}"
            )
            emit({"event": "train", "step": completed_steps, "loss": final_train_loss,
                  "ntp_loss": float(ntp_total),
                  "lr": learning_rate, "grad_norm": float(grad_norm), "tokens_per_second": throughput,
                  "interval_seconds": elapsed, "interval_steps": interval_steps, "peak_cuda_gib": memory,
                  "loop_count": depth or config.loop_count})

        if do_eval:
            synchronize()
            paused_at = time.perf_counter()
            validation_loss = evaluate(
                model,
                val_data,
                batch_size=config.batch_size,
                context_length=config.context_length,
                device=device,
                num_batches=config.eval_batches,
                amp_dtype=amp_dtype,
                seed=config.validation_seed,
            )
            print(f"step={completed_steps} val_loss={validation_loss:.4f}")
            synchronize()
            evaluation_seconds += time.perf_counter() - paused_at
            if not math.isfinite(validation_loss):
                raise RuntimeError("Non-finite validation loss")
            final_loss = validation_loss
            emit({"event": "validation", "step": completed_steps, "loss": validation_loss})

        if do_save:
            synchronize()
            paused_at = time.perf_counter()
            save(completed_steps)
            checkpoint_seconds += time.perf_counter() - paused_at
        if boundary:
            interval_start = time.perf_counter()
            interval_steps = 0
            interval_first_step = completed_steps

    # A fully trained checkpoint can still be evaluated without taking another step.
    if final_loss is None:
        paused_at = time.perf_counter()
        final_loss = evaluate(model, val_data, config.batch_size, config.context_length, device,
                              config.eval_batches, amp_dtype, seed=config.validation_seed)
        evaluation_seconds += time.perf_counter() - paused_at
    summary = {"parameters": sum(p.numel() for p in model.parameters()),
               "inference_parameters": sum(p.numel() for p in model.parameters()),
               "architecture": model.architecture, "unique_layers": len(model.layers),
               "effective_depth": model.effective_depth,
               "start_step": start_step, "completed_steps": config.total_steps,
               "tokens_seen": config.total_steps * tokens_per_step,
               "stable_steps": stable_steps, "stable_training_seconds": stable_seconds,
               "stable_tokens_per_second": stable_steps * tokens_per_step / stable_seconds if stable_seconds else None,
               "training_seconds_including_compile": training_seconds,
               "first_step_seconds_including_compile": first_step_seconds,
               "evaluation_seconds": evaluation_seconds, "checkpoint_seconds": checkpoint_seconds,
               "train_function_seconds": time.perf_counter() - wall_start,
               "peak_cuda_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0,
               "final_train_loss": final_train_loss, "final_validation_loss": final_loss,
               "config": metadata}
    emit({"event": "completed", **summary})
    return summary


def main(argv: list[str] | None = None, on_checkpoint=None) -> dict:
    """The same entry point runs on a laptop or a GPU server."""
    args = vars(parse_args(argv))
    recipe_path = args.pop("config")
    args.pop("output_dir")
    tokenizer_dir = args.pop("tokenizer_dir")
    torch.set_num_threads(args.pop("num_threads"))
    config = TrainingConfig(**args)
    root = config.checkpoint_path.parent
    if config.checkpoint_path.exists() and config.resume_from is None:
        raise ValueError(f"Checkpoint already exists: {config.checkpoint_path}; choose a new output or --resume-from")
    if tokenizer_dir is not None:
        from code_lm.tokenizer import Tokenizer
        tokenizer = Tokenizer.from_files(tokenizer_dir / "vocab.json", tokenizer_dir / "merges.txt", ["<|endoftext|>"])
        if len(tokenizer.vocab) != config.vocab_size:
            raise ValueError("Tokenizer vocabulary does not match model config")
    root.mkdir(parents=True, exist_ok=True)
    effective_config = {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()}
    (root / "config.json").write_text(json.dumps(effective_config, indent=2) + "\n")
    if tokenizer_dir is not None:
        for name in ("vocab.json", "merges.txt", "metadata.json"):
            source = tokenizer_dir / name
            if source.exists() and source.resolve() != (root / name).resolve():
                shutil.copy2(source, root / name)
    summary = train(config, on_checkpoint=on_checkpoint)
    summary.update({"status": "completed", "torch_version": torch.__version__,
                    "gpu": torch.cuda.get_device_name(config.device) if torch.device(config.device).type == "cuda" else None,
                    "recipe": str(recipe_path) if recipe_path is not None else None})
    (root / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    # Export inference weights independently of the optimizer and cloud storage.
    checkpoint = torch.load(config.checkpoint_path, map_location="cpu", weights_only=True)
    model_args = model_args_from_config(asdict(config))
    weights, model_args = inference_weights(checkpoint['model'], model_args)
    torch.save({"model": weights, "model_args": model_args,
                "training_objective": 'ntp',
                "tokens_seen": summary["tokens_seen"]}, root / "model.pt")
    print(json.dumps(summary, indent=2, allow_nan=False))
    return summary


if __name__ == "__main__":
    main()
