# A Small Language Model for Binary Search Tree Code Generation

## 1. Project Overview

We built a small language model in PyTorch to generate Python solutions for binary search tree (BST) problems, and measured how far it generalizes to combinations of task conditions it was never taught.

The project combines a handwritten Transformer, Python pretraining, supervised fine-tuning (SFT), reinforcement learning on execution rewards (GRPO), architectural variants, and execution-based evaluation in an isolated sandbox.

## 2. Model Architecture

Our base model is a decoder-only Transformer with 12 layers, 12 attention heads, a hidden size of 768, and approximately 97.5 million parameters.

We implemented its attention, rotary position embeddings (RoPE), RMSNorm, and SwiGLU components rather than loading a pretrained language model.

The Loop variant reuses eight middle layers twice, producing an effective depth of 20 layers while retaining 12 distinct sets of layer parameters. It has the same parameter count as Base and about 1.7 times its forward compute.

The Loop+LoRA variant adds separate rank-32 LoRA adapters to the query and value projections in each loop iteration, allowing the two iterations to specialize while sharing the main weights.

```mermaid
flowchart TB
    subgraph Base[Base Transformer]
        direction LR
        B0[Token embeddings] --> B1[12 distinct layers] --> B2[Normalization and output head]
    end
    subgraph Loop[Loop Transformer]
        direction LR
        L0[Token embeddings] --> L1[2 initial layers] --> L2[8 shared layers: iteration 1] --> L3[Same 8 layers: iteration 2] --> L4[2 final layers] --> L5[Normalization and output head]
    end
    subgraph Final[Loop Transformer with LoRA]
        direction LR
        F0[Token embeddings] --> F1[2 initial layers] --> F2[Shared core + iteration-1 LoRA] --> F3[Shared core + iteration-2 LoRA] --> F4[2 final layers] --> F5[Normalization and output head]
    end
```

Loop+LoRA has approximately 99.1 million parameters; SFT and RL update both the backbone and the adapters.

## 3. Pretraining

Pretraining data comes from the Python-Edu subset of [HuggingFaceTB/SmolLM-Corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus).

The recorded corpus contains 1,027,740 training documents and approximately 498 million training tokens, with a separate validation corpus of approximately 55 million tokens.

Our data-processing scripts filter unsuitable code, check Python syntax, remove duplicates, separate repositories between splits, and convert the remaining code into token sequences.

We use an 8,192-token BPE vocabulary and a context length of 1,024 tokens.

Base and Loop are pretrained from random initialization to predict the next token for 15,194 steps (about 5 tokens per parameter), using AdamW, learning-rate warmup, and cosine decay. Their final validation losses are 1.412 (Base) and 1.366 (Loop).

## 4. BST Dataset

We generated the BST dataset with our own scripts, using 17 operation contracts, reference solutions, and executable test cases.

The operations include search, minimum and maximum, insertion, predecessor and successor, search paths, and range queries.

Each prompt states the task as Python-style specification fields (for example `operation = 'range_sum'`, `lower_closed = False`, `parity = 'odd'`), followed by the provided `TreeNode` class and the function signature; the target is a Python function.

Every operation has single-condition variants, which are all taught. Its two-condition combinations, 31 in total, are split three ways: 11 are taught in SFT, 10 appear only in RL, and 10 are never trained on. SFT uses 536 examples: 8 per taught specification, with varied function names and two implementation styles.

The test splits measure increasingly distant generalization:

| Split | Tasks | Contents |
| --- | ---: | --- |
| Seen | 67 | every taught specification, with an unseen function name |
| Shuffled | 67 | the same tasks and tests, with the specification fields in a shuffled order |
| Renamed | 67 | the same tasks and tests, with renamed function parameters |
| RL-only pairs | 10 | combinations trained by RL but not SFT |
| Unseen pairs | 10 | combinations never trained on |
| Holdout | 27 | every range specification combining three or four conditions |
| Transfer | 8 | operations absent from training (for example k-th smallest, tree height, lowest common ancestor) |

The generator checks that unseen specifications never reach training, that every reference passes its own tests, and that no taught reference passes the tests of an unseen task.

## 5. Training

**SFT.** Every model is fine-tuned for six epochs with an effective batch of 16 (204 optimizer steps), a learning rate of 0.0001 with warmup and cosine decay, and loss on answer tokens only; the checkpoint with the lowest validation loss is kept. LoRA adapters use the same learning rate as the backbone.

**RL.** We apply Group Relative Policy Optimization (GRPO) after SFT. For each prompt, the model samples eight solutions, each receiving a reward of 1 if it passes all execution tests and 0 otherwise; rewards are normalized within each group to estimate advantages. The objective combines a clipped policy loss with a KL penalty against the SFT model. Training runs 100 rounds of 16 prompts with a learning rate of 0.00001, clipping of 0.2, and a KL coefficient of 0.02. Before training, prompts that the SFT model solves in all or none of eight samples are removed, and each prompt is paired with one that differs in a single condition. The final checkpoint is chosen by validation pass rate.

**No pretraining.** As a control, Base and Loop are fine-tuned from the same random initialization used for pretraining, for 30 epochs.

**Qwen baseline.** Qwen2.5-Coder-0.5B-Instruct receives each task as an English translation of the same specification. It is evaluated zero-shot, and after SFT on the same 536 examples with the same recipe.

## 6. vLLM Inference

We implemented a custom vLLM adapter so that our handwritten model can use vLLM scheduling and KV caching; RL sampling uses it.

Before inference, the exporter unfolds the shared loop into separate execution layers and merges each iteration's LoRA weights into its query and value projections.

## 7. Evaluation and Results

A sample passes only when its generated function passes every test in its execution suite. Each task receives eight samples at temperature 1, and the table reports pass@1, the mean pass rate over samples. Model output is executed only in a network-blocked sandbox.

| Model | Seen | Shuffled | Renamed | RL-only pairs | Unseen pairs | Holdout | Transfer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base, no pretraining (SFT, 30 epochs) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Loop, no pretraining (SFT, 30 epochs) | 8.6 | 7.8 | 0.0 | 8.8 | 2.5 | 4.6 | 0.0 |
| Base + SFT | 76.5 | 68.3 | 52.8 | 66.2 | 67.5 | 31.9 | 0.0 |
| Base + SFT + RL | 88.6 | 76.5 | 59.0 | 96.2 | 85.0 | 73.6 | 0.0 |
| Loop + SFT | 91.0 | 72.8 | 63.1 | 92.5 | 87.5 | 63.0 | 0.0 |
| Loop + SFT + RL | 94.6 | 70.7 | 64.2 | 97.5 | 95.0 | 85.2 | 0.0 |
| Loop + LoRA + SFT | 92.2 | 73.7 | 63.6 | 91.2 | 87.5 | 66.2 | 0.0 |
| Loop + LoRA + SFT + RL | 94.4 | 73.7 | 69.6 | 95.0 | 96.2 | 95.8 | 0.0 |
| Qwen2.5-Coder-0.5B-Instruct, zero-shot | 4.5 | — | — | 0.0 | 0.0 | 0.0 | 23.4 |
| Qwen2.5-Coder-0.5B-Instruct + SFT | 94.6 | — | — | 96.2 | 96.2 | 94.4 | 0.0 |

pass@1 (%), 8 samples per task at temperature 1. The shuffled and renamed splits do not apply to Qwen, whose English prompts have no field order or parameter names.
