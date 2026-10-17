"""Generate BST solutions with our own Transformer checkpoint (no code execution)."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import torch

from code_lm.inference.torch_generate import generate_batch
from code_lm.training.sft import load_model
from code_lm.data.prepare import load_tokenizer, read_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/bst.pt"))
    parser.add_argument("--tokenizer-dir", type=Path, default=Path("data/tokenizer"))
    parser.add_argument("--data", type=Path, default=Path("data/bst_covered"), help="Evaluation JSONL or directory containing both test splits")
    parser.add_argument("--output", type=Path, default=Path("runs/generations.jsonl"))
    parser.add_argument("--spec", type=Path, help="JSON object with spec and optional entry_point; bypass natural-language parsing")
    parser.add_argument("--task-id", help="Generate one dataset task for a demo")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--backend", choices=["torch", "vllm"], default="torch")
    parser.add_argument("--eager", action="store_true", help="Disable vLLM CUDA Graphs for initial setup or debugging")
    args = parser.parse_args()
    if args.eager and args.backend != "vllm":
        parser.error("--eager requires --backend vllm")
    if args.backend == "vllm":
        from code_lm.inference.generate import check_runtime
        check_runtime()
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}. Supply --checkpoint or place weights at checkpoints/bst.pt.")
    if args.batch_size < 1 or args.max_new_tokens < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Batch size, output budget and limit must be positive")
    if args.output.exists():
        raise FileExistsError(f"Choose a new output path: {args.output}")
    if args.spec:
        if args.task_id:
            parser.error("--spec cannot be combined with --task-id")
        from code_lm.data.bst_spec import render_spec
        supplied = json.loads(args.spec.read_text())
        rendered = render_spec(supplied["spec"], supplied.get("entry_point", "solve"))
        rows = [{"id": "spec-" + hashlib.sha256(rendered.encode()).hexdigest()[:16], "prompt": rendered}]
    else:
        rows = list(read_rows(args.data))
    if args.task_id:
        rows = [r for r in rows if r["id"] == args.task_id]
    rows = rows[:args.limit]
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Expected nonempty tasks with unique IDs")
    tokenizer = load_tokenizer(args.tokenizer_dir)
    if args.backend == "torch":
        model, config, _ = load_model(args.checkpoint, torch.device(args.device))
        model.eval()
    else:
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        metadata = saved.get("config") or {}
        config = saved.get("model_args") or metadata.get("model_args") or metadata
        del saved
    if set(tokenizer.vocab) != set(range(config["vocab_size"])):
        raise ValueError("Tokenizer vocabulary does not match checkpoint")
    eos_id = tokenizer.token_to_id[b"<|endoftext|>"]
    prompts = [tokenizer.encode(row["prompt"]) for row in rows]
    if any(not p or len(p) + args.max_new_tokens > config["context_length"] for p in prompts):
        raise ValueError("Prompt plus output budget exceeds context; lower --max-new-tokens")
    with args.checkpoint.open("rb") as stream:
        checkpoint_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    vllm_outputs = None
    if args.backend == "vllm":
        from code_lm.inference import register
        from code_lm.inference.export import export_checkpoint
        from code_lm.inference.generate import engine_options, sampling_options
        from vllm import LLM, SamplingParams

        register()
        with tempfile.TemporaryDirectory(prefix="bst-vllm-") as directory:
            exported = Path(directory) / "model"
            export_checkpoint(args.checkpoint, exported, eos_id)
            llm = LLM(**engine_options(exported, config["context_length"], eager=args.eager))
            requests = llm.generate([{"prompt_token_ids": p} for p in prompts],
                                    SamplingParams(**sampling_options(eos_id, args.max_new_tokens)),
                                    use_tqdm=False)
            vllm_outputs = []
            for request in requests:
                completion = request.outputs[0]
                ids = list(completion.token_ids)
                if ids and ids[-1] == eos_id:
                    ids.pop()
                vllm_outputs.append({"token_ids": ids, "finish_reason": completion.finish_reason})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        for start in range(0, len(rows), args.batch_size):
            outputs = (vllm_outputs[start:start + args.batch_size] if vllm_outputs is not None else
                       generate_batch(model, prompts[start:start + args.batch_size],
                                     max_new_tokens=args.max_new_tokens, eos_id=eos_id,
                                     context_length=config["context_length"]))
            for row, generated in zip(rows[start:start + args.batch_size], outputs, strict=True):
                result = {"id": row["id"], "completion": tokenizer.decode(generated["token_ids"]),
                          "finish_reason": generated["finish_reason"],
                          "backend": args.backend,
                          "architecture": config.get("architecture", "standard"),
                          "loop_count": config.get("loop_count", 1),
                          "checkpoint_sha256": checkpoint_hash,
                          "prompt_sha256": hashlib.sha256(row["prompt"].encode()).hexdigest()}
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                if len(rows) == 1:
                    print(row["prompt"] + result["completion"])
            stream.flush()
            print(f"Generated {min(start + args.batch_size, len(rows))}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
