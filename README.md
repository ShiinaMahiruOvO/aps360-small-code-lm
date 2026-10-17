# Usage

Run commands from the repository root. Use Python 3.12 or 3.13. Supply a trained checkpoint with `--checkpoint`; weights and the full Python pretraining corpus are not bundled.

## Install

```bash
uv sync --frozen
```

## Prepare data

```bash
# Verify the supplied BST dataset.
uv run python prepare.py bst
# Regenerate it in a new directory.
uv run python prepare.py bst --regenerate --output data/generated/bst
# Download and tokenize a small Python-Edu sample.
uv run python prepare.py python-edu --sample-size 1000 --output-dir data/python-edu
uv run python prepare.py tokens --source data/python-edu
```

The small sample is for trying the pipeline. Full pretraining requires the larger token arrays at the paths in `configs/pretrain.json`; reduce the configured training budget when using a small sample.

## Train

```bash
uv run python pretrain.py --config configs/pretrain.json
uv run python sft.py --config configs/sft.json
```

SFT writes its selected checkpoint to `runs/sft/loop_lora/best.pt`.

## Infer with PyTorch

```bash
# Generate one answer on a local NVIDIA GPU; use --device cpu for CPU inference.
uv run python generate.py --checkpoint runs/sft/loop_lora/best.pt \
  --device cuda --limit 1 --output runs/demo.jsonl
# Generate answers for all 87 evaluation tasks.
uv run python generate.py --checkpoint runs/sft/loop_lora/best.pt \
  --device cuda --output runs/answers.jsonl
```

To supply a structured task instead of a dataset task, replace `--limit 1` with `--spec configs/bst_spec_example.json`. Use a new output filename on each run.

## Infer locally with vLLM

Use Linux (or Linux under WSL2), an NVIDIA Ampere-or-newer GPU, and a working NVIDIA driver. This project's vLLM backend uses CUDA/BF16. See the [vLLM installation requirements](https://docs.vllm.ai/en/v0.18.0/getting_started/installation/gpu/).

Create a separate environment so vLLM can install its compatible PyTorch and NumPy versions:

```bash
nvidia-smi
uv venv .venv-vllm --python 3.12
uv pip install --python .venv-vllm/bin/python -e . \
  'vllm==0.18.0' 'transformers==4.57.6'
uv pip check --python .venv-vllm/bin/python

# First run: one task, without CUDA Graphs.
.venv-vllm/bin/python generate.py --backend vllm --eager \
  --checkpoint runs/sft/loop_lora/best.pt --limit 1 --output runs/vllm-demo.jsonl
# Full evaluation generation, with CUDA Graphs enabled.
.venv-vllm/bin/python generate.py --backend vllm \
  --checkpoint runs/sft/loop_lora/best.pt --output runs/vllm-answers.jsonl
```

Use `.venv-vllm/bin/python` directly for these commands; `uv run` selects the separate training environment. The entry point automatically exports the checkpoint, expands the loop, merges its LoRA weights and registers the custom vLLM model. No separate export or cloud service is required. Add `--eager` to disable CUDA Graphs if needed.

## Evaluate

```bash
uv run python evaluate.py --generations runs/answers.jsonl --export-jobs runs/jobs.json
```

Copy `runs/jobs.json` to `/tmp/input.json` inside an already isolated, network-blocked, credential-free Linux environment. Install Python 3.13 and `pytest==9.0.2` there, copy `code_lm/evaluation/sandbox.py`, and run the following **inside that isolated environment** as root; its worker processes run as separate unprivileged users:

```bash
BST_ISOLATED_GRADER=1 python sandbox.py
```

Copy `/tmp/results.jsonl` back to `runs/results.jsonl`, then score it:

```bash
uv run python evaluate.py --generations runs/answers.jsonl \
  --sandbox-results runs/results.jsonl --output runs/scores.json
```

For answers generated with vLLM, use `runs/vllm-answers.jsonl` in both evaluation commands. Do not execute generated answers on the development host.

## Generate Qwen comparison answers

Use the Linux CUDA inference environment:

```bash
.venv-vllm/bin/python -m code_lm.evaluation.qwen --kind base --output runs/qwen-base
.venv-vllm/bin/python -m code_lm.evaluation.qwen --kind instruct --output runs/qwen-instruct
```

## Reinforcement learning

Provide an external grader bridge that submits jobs to the isolated evaluation environment described above and copies its results back; `grader_bridge.py` below is a user-supplied bridge, not a bundled script.

```bash
.venv-vllm/bin/python rl.py --checkpoint runs/sft/loop_lora/best.pt \
  --config configs/rl.json --backend vllm --device cuda --output runs/rl \
  --grader-command 'python grader_bridge.py --input {input} --output {output}'
```

The bridge receives a JSON array of grading jobs and must write the sandbox results as JSONL. For PyTorch sampling, use `uv run python rl.py` with `--backend torch` and the appropriate device. Add `--resume` to continue the same run with unchanged configuration. Start from the unmerged SFT checkpoint; the reference remains frozen, while GRPO updates the backbone and adapters. Results are written to `runs/rl/comparison.json` and validation progress to `runs/rl/curves.json`.

## Evaluate the compositional holdout

The frozen `data/bst_holdout/test.jsonl` contains 27 untaught range-condition combinations. Generate answers with the same inference commands, adding `--data data/bst_holdout/test.jsonl`. For both grading commands, add `--data data/bst_holdout/test.jsonl --suites data/bst_holdout/suites.json`.

To reproduce the dataset and its trusted-reference overlap audit in a new directory:

```bash
uv run python -m code_lm.data.bst_holdout --output data/generated/holdout
```

Keep this split out of SFT, RL, and checkpoint selection.
