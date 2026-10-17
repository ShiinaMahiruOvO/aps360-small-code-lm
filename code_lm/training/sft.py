"""Supervised fine-tuning: load a base model and learn from reference answers."""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import shutil

import numpy as np
import torch
from torch.nn import functional as F

from code_lm.training.checkpoint import load_checkpoint, save_checkpoint
from code_lm.model import TransformerLM, model_args_from_config, inference_weights
from code_lm.data.sft import IGNORE_INDEX, SFTDataset, collate_examples
from code_lm.training.schedule import cosine_learning_rate_schedule
from code_lm.tokenizer import Tokenizer
from code_lm.training.common import autocast_context


@dataclass
class SFTConfig:
    checkpoint: Path
    tokenizer_dir: Path
    train_data: Path
    val_data: Path
    output_dir: Path
    epochs: int = 3
    batch_size: int = 4
    grad_accum_steps: int = 4
    learning_rate: float = 2e-5
    min_learning_rate: float = 2e-6
    warmup_ratio: float = 0.03
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    device: str = "cpu"
    precision: str = "auto"
    seed: int = 42
    eval_every: int = 50
    log_every: int = 10
    resume_from: Path | None = None
    loop_max_count: int = 0
    loop_lora_rank: int = 0
    loop_lora_alpha: float = 8.0
    loop_lora_learning_rate: float | None = None


def make_optimizer(model, config, device):
    """Keep the legacy two groups unless an adapter-specific rate is requested."""
    groups = []
    separate = config.loop_lora_learning_rate is not None
    for adapter in ((False, True) if separate else (False,)):
        scale = config.loop_lora_learning_rate / config.learning_rate if adapter else 1.
        for decay in (True, False):
            params = [p for name, p in model.named_parameters()
                      if (p.ndim >= 2) == decay
                      and (not separate or name.startswith('loop_adapters.') == adapter)]
            if params:
                groups.append({'params': params, 'weight_decay': config.weight_decay if decay else 0.,
                               'lr': config.learning_rate * scale, 'lr_scale': scale})
    return torch.optim.AdamW(groups, lr=config.learning_rate, betas=(0.9, 0.95),
                             fused=device.type == 'cuda')


def configure_loop(model, args, config):
    """Extend a pretrained loop with optional zero-initialized per-depth adapters."""
    if not config.loop_max_count and not config.loop_lora_rank:
        return model, args
    wanted = {**args, 'loop_max_count': config.loop_max_count,
              'loop_lora_rank': config.loop_lora_rank, 'loop_lora_alpha': config.loop_lora_alpha}
    if config.resume_from:
        if args != wanted:
            raise ValueError('Resume cannot change loop architecture')
        return model, args
    if args.get('loop_lora_rank', 0):
        if args != wanted:
            raise ValueError('Cannot change existing adapter architecture')
        return model, args
    upgraded = TransformerLM(**wanted).to(next(model.parameters()).device)
    incompatible = upgraded.load_state_dict(model.state_dict(), strict=False)
    if incompatible.unexpected_keys or any(not k.startswith('loop_adapters.') for k in incompatible.missing_keys):
        raise ValueError('Unexpected architecture difference during adapter initialization')
    return upgraded, wanted


def answer_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Sum answer-token losses; the caller supplies the appropriate denominator."""
    return F.cross_entropy(
        logits.float().flatten(0, 1), labels.flatten(),
        ignore_index=IGNORE_INDEX, reduction="sum",
    )


def train_step(model, optimizer, examples, batch_size, pad_id, device, amp_dtype, max_grad_norm,
               loop_count=None):
    """One optimizer update, possibly split into several smaller forward passes."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    num_tokens = sum(int((labels != IGNORE_INDEX).sum()) for _, labels in examples)
    if num_tokens == 0:
        raise ValueError("The update contains no answer tokens")

    total_loss = torch.zeros((), device=device)
    for start in range(0, len(examples), batch_size):
        inputs, labels = collate_examples(examples[start:start + batch_size], pad_id)
        inputs, labels = inputs.to(device), labels.to(device)
        with autocast_context(device, amp_dtype):
            loss_sum = answer_loss(model(inputs, loop_count=loop_count), labels)
        # Divide by the answer-token count of the WHOLE update. Averaging each
        # microbatch separately would give short answers too much weight.
        (loss_sum / num_tokens).backward()
        total_loss += loss_sum.detach()

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    return float(total_loss / num_tokens), float(grad_norm), num_tokens


@torch.no_grad()
def evaluate(model, dataset, batch_size, device, amp_dtype):
    """Mean answer-token loss over the entire validation set."""
    was_training = model.training
    model.eval()
    total_loss = torch.zeros((), device=device)
    num_tokens = 0
    try:
        for start in range(0, len(dataset), batch_size):
            inputs, labels = collate_examples(dataset.examples[start:start + batch_size], dataset.eos_id)
            inputs, labels = inputs.to(device), labels.to(device)
            with autocast_context(device, amp_dtype):
                total_loss += answer_loss(model(inputs), labels)
            num_tokens += int((labels != IGNORE_INDEX).sum())
    finally:
        model.train(was_training)
    loss = float(total_loss / num_tokens)
    if not math.isfinite(loss):
        raise RuntimeError("Non-finite validation loss")
    return loss


def load_model(path: Path, device: torch.device):
    """Accept either our inference export or a full pretraining/SFT checkpoint."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    config = saved.get("config") or {}
    source = saved.get("model_args") or config.get("model_args") or config
    model_args = model_args_from_config(source)
    weights, model_args = inference_weights(saved['model'], model_args)
    model = TransformerLM(**model_args)
    model.load_state_dict(weights, strict=True)
    return model.to(device), model_args, saved


def train(config: SFTConfig) -> dict:
    for name in ("epochs", "batch_size", "grad_accum_steps", "eval_every", "log_every"):
        if getattr(config, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if not 0 <= config.warmup_ratio < 1:
        raise ValueError("warmup_ratio must be in [0, 1)")
    if not 0 <= config.min_learning_rate <= config.learning_rate or config.learning_rate <= 0:
        raise ValueError("Require 0 <= min_learning_rate <= learning_rate and learning_rate > 0")
    if config.loop_lora_learning_rate is not None:
        if not math.isfinite(config.loop_lora_learning_rate) or config.loop_lora_learning_rate <= 0:
            raise ValueError('Adapter learning rate must be finite and positive')
        if not config.loop_lora_rank:
            raise ValueError('Adapter learning rate requires loop LoRA')
    if config.max_grad_norm <= 0 or config.weight_decay < 0:
        raise ValueError("max_grad_norm must be positive and weight_decay nonnegative")
    if config.precision not in ("auto", "fp32", "bf16"):
        raise ValueError("precision must be auto, fp32 or bf16")
    root = config.output_dir
    if config.resume_from is None and any((root / name).exists() for name in ("latest.pt", "model.pt", "metrics.jsonl")):
        raise ValueError(f"Run already exists: {root}; use a new output directory or --resume-from")
    if config.resume_from is not None and config.resume_from.resolve() != (root / "latest.pt").resolve():
        raise ValueError("Resume from output_dir/latest.pt; use --checkpoint to start a new training phase")

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    bf16 = config.precision == "bf16" or (
        config.precision == "auto" and device.type == "cuda" and torch.cuda.is_bf16_supported()
    )
    if bf16 and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("Use fp32 on this device; this trainer enables BF16 on supported CUDA GPUs")
    amp_dtype = torch.bfloat16 if bf16 else None

    model, model_args, saved = load_model(config.resume_from or config.checkpoint, device)
    model, model_args = configure_loop(model, model_args, config)
    tokenizer = Tokenizer.from_files(
        config.tokenizer_dir / "vocab.json", config.tokenizer_dir / "merges.txt", ["<|endoftext|>"],
    )
    if set(tokenizer.vocab) != set(range(model_args["vocab_size"])):
        raise ValueError("Tokenizer vocabulary does not match checkpoint vocabulary")
    train_data = SFTDataset(config.train_data, tokenizer, model_args["context_length"])
    val_data = SFTDataset(config.val_data, tokenizer, model_args["context_length"])
    if config.train_data.resolve() == config.val_data.resolve() or train_data.group_ids & val_data.group_ids:
        raise ValueError("Training and validation data must be separate, with no overlapping groups")

    effective_batch_size = config.batch_size * config.grad_accum_steps
    steps_per_epoch = math.ceil(len(train_data) / effective_batch_size)
    total_steps = steps_per_epoch * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    run_config = {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()}
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in {
        "train": config.train_data, "validation": config.val_data,
        "vocab": config.tokenizer_dir / "vocab.json", "merges": config.tokenizer_dir / "merges.txt",
    }.items()}
    metadata = {"stage": "sft", "model_args": model_args, "run_config": run_config, "hashes": hashes}

    # SFT is a new objective. Start a fresh optimizer; only an SFT resume restores it.
    optimizer = make_optimizer(model, config, device)
    start_step, tokens_seen = 0, 0
    best_val_loss = float("inf")
    if config.resume_from is not None:
        previous = saved.get("config") or {}
        mutable = {"resume_from", "device", "log_every", "eval_every"}
        if previous.get("stage") != "sft" or previous.get("hashes") != hashes:
            raise ValueError("Resume requires an SFT checkpoint with unchanged data and tokenizer")
        new_defaults = {'loop_max_count': 0, 'loop_lora_rank': 0, 'loop_lora_alpha': 8.0,
                        'loop_lora_learning_rate': None}
        changed = [k for k, v in run_config.items() if k not in mutable and previous["run_config"].get(k, new_defaults.get(k)) != v]
        if changed:
            raise ValueError(f"Resume configuration differs: {changed}")
        start_step = load_checkpoint(config.resume_from, model, optimizer)
        tokens_seen = saved["tokens_seen"]
        best_val_loss = previous["best_validation_loss"]
    if not 0 <= start_step <= total_steps:
        raise ValueError("Checkpoint step is outside the configured training schedule")
    del saved

    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for name in ("vocab.json", "merges.txt"):
        source = config.tokenizer_dir / name
        if source.resolve() != (root / name).resolve():
            shutil.copy2(source, root / name)

    def log(record):
        print(json.dumps(record, allow_nan=False), flush=True)
        with (root / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")

    def export(path):
        torch.save({"model": model.state_dict(), "model_args": model_args,
                    "stage": "sft", "tokens_seen": tokens_seen}, path)

    initial_val_loss = evaluate(model, val_data, config.batch_size, device, amp_dtype)
    val_loss = initial_val_loss
    log({"event": "start", "step": start_step, "validation_loss": val_loss,
         "train_examples": len(train_data), "validation_examples": len(val_data), "total_steps": total_steps})
    if start_step == 0:
        best_val_loss = val_loss
        export(root / "best.pt")

    # Every epoch has a reproducible shuffle. A resumed run skips completed
    # updates in that same order, including the short final batch of each epoch.
    for epoch in range(start_step // steps_per_epoch, config.epochs):
        order = list(range(len(train_data)))
        random.Random(config.seed + epoch).shuffle(order)
        for batch_index in range(steps_per_epoch):
            step = epoch * steps_per_epoch + batch_index
            if step < start_step:
                continue
            offset = batch_index * effective_batch_size
            examples = [train_data[i] for i in order[offset:offset + effective_batch_size]]
            lr = cosine_learning_rate_schedule(
                step + 1, config.learning_rate, config.min_learning_rate, warmup_steps, total_steps,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr * group.get('lr_scale', 1.)
            depth = model.loop_count
            loss, grad_norm, num_tokens = train_step(
                model, optimizer, examples, config.batch_size, train_data.eos_id, device, amp_dtype,
                config.max_grad_norm, loop_count=depth,
            )
            tokens_seen += num_tokens
            completed_steps = step + 1
            if completed_steps == 1 or completed_steps % config.log_every == 0 or completed_steps == total_steps:
                log({"event": "train", "step": completed_steps, "epoch": epoch + 1,
                     "answer_loss": loss, "lr": lr, "grad_norm": grad_norm, "answer_tokens": tokens_seen,
                     "adapter_lr": (lr * config.loop_lora_learning_rate / config.learning_rate
                                    if config.loop_lora_learning_rate is not None else None),
                     "loop_count": depth or model.loop_count})

            if completed_steps % config.eval_every == 0 or batch_index == steps_per_epoch - 1:
                val_loss = evaluate(model, val_data, config.batch_size, device, amp_dtype)
                log({"event": "validation", "step": completed_steps, "answer_loss": val_loss})
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    export(root / "best.pt")
                save_checkpoint(model, optimizer, completed_steps, root / "latest.pt",
                                config={**metadata, "best_validation_loss": best_val_loss}, tokens_seen=tokens_seen)

    export(root / "model.pt")
    summary = {"stage": "sft", "status": "completed", "start_step": start_step, "completed_steps": total_steps,
               "train_examples": len(train_data), "validation_examples": len(val_data),
               "answer_tokens_seen": tokens_seen, "initial_validation_loss": initial_val_loss,
               "final_validation_loss": val_loss, "best_validation_loss": best_val_loss}
    depths = [model.loop_count] * total_steps
    summary.update(depth_counts={str(k): depths.count(k) for k in sorted(set(depths))},
                   mean_loop_count=sum(depths)/len(depths),
                   parameters=sum(p.numel() for p in model.parameters()),
                   validation_loop_count=model.loop_count)
    (root / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    log({"event": "completed", **summary})
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    for name in ("checkpoint", "tokenizer-dir", "train-data", "val-data", "output-dir", "resume-from"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--min-learning-rate", type=float, default=2e-6)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=("auto", "fp32", "bf16"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument('--loop-max-count', type=int, default=0)
    parser.add_argument('--loop-lora-rank', type=int, default=0)
    parser.add_argument('--loop-lora-alpha', type=float, default=8.0)
    parser.add_argument('--loop-lora-learning-rate', type=float)
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--config", type=Path)
    recipe_path = probe.parse_known_args(argv)[0].config
    if recipe_path is not None:
        recipe = json.loads(recipe_path.read_text())
        actions = {a.dest: a for a in parser._actions if a.dest not in {"help", "config"}}
        if not isinstance(recipe, dict) or recipe.keys() - actions.keys():
            parser.error("Config must be an object containing only recognized CLI options")
        for key, value in recipe.items():
            action = actions[key]
            if value is not None and action.type is not None:
                value = action.type(value)
            if action.choices is not None and value not in action.choices:
                parser.error(f"Invalid {key}: {value}")
            recipe[key] = value
        parser.set_defaults(**recipe)
    args = parser.parse_args(argv)
    for name in ("checkpoint", "tokenizer_dir", "train_data", "val_data", "output_dir"):
        if getattr(args, name) is None:
            parser.error(f"{name} must be supplied in the config or CLI")
    if args.num_threads <= 0:
        parser.error("num_threads must be positive")
    return args


def main(argv=None):
    args = vars(parse_args(argv))
    args.pop("config")
    torch.set_num_threads(args.pop("num_threads"))
    return train(SFTConfig(**args))


if __name__ == "__main__":
    main()
