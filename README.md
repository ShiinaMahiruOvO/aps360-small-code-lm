# A 100M-Parameter Model for Simple Python Programming

This APS360 project studies whether a decoder-only Transformer with roughly 100 million parameters, pretrained from scratch on educational Python code, can learn a family of binary-search-tree programming tasks, and how far it generalizes to combinations of conditions it was never taught. Functional correctness is measured by executing hidden unit tests in an isolated sandbox.

Three handwritten architectures share the same parameter count: a 12-layer Transformer (Base), a looped Transformer that runs its 8 middle layers twice (Loop), and Loop with per-iteration rank-32 LoRA adapters on the query and value projections (Loop+LoRA). Each is pretrained on Python-Edu, fine-tuned on the BST tasks, and then trained with GRPO on execution rewards. A Qwen2.5-Coder-0.5B-Instruct baseline is fine-tuned on the same tasks.

## Usage

Run commands from the repository root. Use Python 3.12 or 3.13. Supply a trained checkpoint with `--checkpoint`; weights and the full Python pretraining corpus are not bundled.

## Install

```bash
uv sync --frozen
```

## Data

`data/bst` holds every BST split, prompt and test suite. Each of 17 operations (search, floor, insert, range sum, …) has single-condition variants, which are all taught, and 31 two-condition combinations, which are split three ways: 11 are taught in SFT, 10 appear only in RL, and 10 are never trained on. The test splits form a ladder:

| Split | Tasks | Contents |
|---|---|---|
| `test_seen` | 67 | every specification taught in SFT, with an unseen function name |
| `test_seen_shuffled` | 67 | the same tasks and tests with the specification fields in a shuffled order |
| `test_seen_renamed` | 67 | the same tasks and tests with renamed parameters |
| `test_rl_pair` | 10 | the RL-only combinations |
| `test_unseen_pair` | 10 | combinations never trained on |
| `test_holdout` | 27 | every range specification with 3–4 changed conditions |
| `test_transfer` | 8 | operations absent from training (k-th smallest, height, lowest common ancestor, …) |

`train`, `validation` and `rl` cover the taught specifications; `validation` and `rl` also include the RL-only combinations. Keep `test_unseen_pair`, `test_holdout` and `test_transfer` out of training and checkpoint selection.

```bash
# Verify the dataset: file and suite hashes, and which specifications each split may contain.
uv run python prepare.py bst
# Regenerate it in a new directory (byte-identical); this also runs every reference against its tests
# and checks that no taught reference passes an unseen task's tests.
uv run python prepare.py bst --regenerate --output data/generated/bst
# Download and tokenize a small Python-Edu sample.
uv run python prepare.py python-edu --sample-size 1000 --output-dir data/python-edu
uv run python prepare.py tokens --source data/python-edu
```

The small sample is for trying the pipeline. Full pretraining requires the larger token arrays at the paths in `configs/pretrain/*.json`; reduce the configured training budget when using a small sample.

## Train

```bash
# Pretrain the 12-layer baseline and the looped model.
uv run python pretrain.py --config configs/pretrain/baseline.json
uv run python pretrain.py --config configs/pretrain/loop.json
# SFT each architecture (learning rate 1e-4; LoRA adapters use the backbone learning rate).
uv run python sft.py --config configs/sft/baseline.json
uv run python sft.py --config configs/sft/loop.json
uv run python sft.py --config configs/sft/loop_lora.json
```

Each SFT run keeps the checkpoint with the lowest validation loss at `runs/sft/<run>/best.pt`. `configs/sft/scratch/` fine-tunes Base and Loop without pretraining, for 30 epochs: their `checkpoint` is the architecture file `configs/pretrain/<arch>.json`, from which SFT starts at the same seeded random initialization as pretraining. The commands below use `runs/sft/loop_lora/best.pt` as a placeholder for a checkpoint.

## Infer with PyTorch

```bash
# Generate one answer on a local NVIDIA GPU; use --device cpu for CPU inference.
uv run python generate.py --checkpoint runs/sft/loop_lora/best.pt \
  --device cuda --limit 1 --output runs/demo.jsonl
# Generate greedy answers for all 256 test tasks.
uv run python generate.py --checkpoint runs/sft/loop_lora/best.pt \
  --device cuda --output runs/answers.jsonl
# Draw 8 samples per task at temperature 1 for pass@1 and pass@8.
uv run python generate.py --checkpoint runs/sft/loop_lora/best.pt \
  --device cuda --samples 8 --temperature 1.0 --output runs/samples.jsonl
```

To supply a structured task instead of a dataset task, replace `--limit 1` with `--spec configs/bst_spec_example.json`. To use a single split, pass its file with `--data`, e.g. `--data data/bst/test_holdout.jsonl`. Use a new output filename on each run.

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

The report gives, per split, pass@1 and pass@8 (with 8 samples per task), a task-level bootstrap 95% CI and pass@1 per operation. When grading a single split, add `--data data/bst/<split>.jsonl`. For answers generated with vLLM, use `runs/vllm-answers.jsonl` in both evaluation commands. Do not execute generated answers on the development host.

## Qwen baseline

Qwen2.5-Coder-0.5B-Instruct receives each task as an English translation of the same specification. It is evaluated zero-shot, or after SFT on `data/bst/train.jsonl` with the recipe of `configs/sft`. Use the Linux CUDA inference environment:

```bash
.venv-vllm/bin/python -m code_lm.evaluation.qwen --output runs/qwen-zero-shot
.venv-vllm/bin/python -m code_lm.evaluation.qwen --learning-rate 1e-4 --output runs/qwen-sft
```

Each run writes 8 samples per task for validation and the test splits (`<split>.jsonl`); grade them with `evaluate.py --generations runs/qwen-sft/test_holdout.jsonl ...` as above. The shuffled and renamed splits do not apply, because the English prompt has no field order or parameter names.

## Reinforcement learning

Provide an external grader bridge that submits jobs to the isolated evaluation environment described above and copies its results back; `grader_bridge.py` below is a user-supplied bridge, not a bundled script.

```bash
.venv-vllm/bin/python rl.py --checkpoint runs/sft/loop_lora/best.pt \
  --config configs/rl.json --backend vllm --device cuda --output runs/rl \
  --grader-command 'python grader_bridge.py --input {input} --output {output}'
```

The bridge receives a JSON array of grading jobs and must write the sandbox results as JSONL. `configs/rl.json` runs 100 rounds of 16 prompts × 8 samples at learning rate 1e-5. It first keeps only the RL prompts the SFT model sometimes but not always solves, pairs each prompt with one that differs in a single condition, validates with 8 samples every 20 rounds, and finally selects the best validated checkpoint or the average of the top three. For PyTorch sampling, use `uv run python rl.py` with `--backend torch` and the appropriate device. Add `--resume` to continue the same run with unchanged configuration. Start from the unmerged SFT checkpoint; the reference remains frozen, while GRPO updates the backbone and adapters. Results are written to `runs/rl/comparison.json` and validation progress to `runs/rl/curves.json`.

## Results

`results/scores.json` holds pass@1 and pass@8 on every test split for each model in the report (8 samples per task at temperature 1). `results/curves/` holds the training curves: pretraining loss (`pretrain.csv`), SFT answer loss (`sft.csv`), and the per-round RL rollout pass rate, KL and validation pass@1 (`rl.csv`).
