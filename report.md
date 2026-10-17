# A Small Language Model for Binary Search Tree Code Generation

## 1. Project Overview

We built a small language model in PyTorch to generate Python solutions for binary search tree (BST) problems.

The project combines a handwritten Transformer, Python pretraining, supervised fine-tuning (SFT), architectural improvements, execution-based evaluation, and vLLM inference.

## 2. Model Architecture

Our base model is a decoder-only Transformer with 12 layers, 12 attention heads, a hidden size of 768, and approximately 97.5 million parameters.

We implemented its attention, rotary position embeddings (RoPE), RMSNorm, and SwiGLU components rather than loading a pretrained language model.

The Loop variant reuses eight middle layers twice, producing an effective depth of 20 layers while retaining 12 distinct sets of layer parameters.

The final variant adds separate rank-32 LoRA adapters to the query and value projections in each loop iteration, allowing the two iterations to specialize while sharing the main weights.

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

The final model has approximately 99.1 million parameters, and SFT updates both the backbone and the adapters.

## 3. Pretraining and Data

Pretraining data comes from the Python-Edu subset of [HuggingFaceTB/SmolLM-Corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus).

The recorded corpus contains 1,027,740 training documents and approximately 498 million training tokens, with a separate validation corpus of approximately 55 million tokens.

Our data-processing scripts filter unsuitable code, check Python syntax, remove duplicates, separate repositories between splits, and convert the remaining code into token sequences.

We use an 8,192-token BPE vocabulary and a context length of 1,024 tokens.

The models are pretrained from random initialization to predict the next token, using AdamW, learning-rate warmup, and cosine decay.

## 4. Supervised Fine-Tuning

We generated the BST dataset with our own scripts, using 17 operation contracts, reference solutions, and executable test cases.

The operations include search, minimum and maximum, insertion, predecessor and successor, search paths, and range queries.

Each prompt expresses the task through fixed English specification fields, while its target is a Python function.

The dataset contains 552 training examples, 87 validation examples, and 87 evaluation tasks: 56 basic or single-condition tasks and 31 tasks combining conditions.

SFT runs for six epochs, computes loss only on answer tokens, and selects the checkpoint with the lowest validation answer loss.

For the final model, the initial learning rates are 0.00002 for the backbone and 0.0001 for the rank-32 adapters.

## 5. vLLM Inference

We implemented a custom vLLM adapter so that our handwritten model can use vLLM scheduling and KV caching.

Before inference, the exporter unfolds the shared loop into separate execution layers and merges each iteration's LoRA weights into its query and value projections.

## 6. Evaluation and Results

### Original evaluation dataset

A task passes only when its generated function passes every test in its execution suite.

The evaluation covers all 17 operation types below; counts refer to tasks, not individual execution cases.

| Operation | Requirement | Basic / single condition | Condition combinations |
| --- | --- | ---: | ---: |
| `contains` | Check whether a target exists. | 1 | 0 |
| `absent` | Check whether a target is absent. | 1 | 0 |
| `minimum` | Return the smallest key. | 3 | 0 |
| `maximum` | Return the largest key. | 3 | 0 |
| `floor` | Return the greatest key less than or equal to the target. | 3 | 0 |
| `ceiling` | Return the smallest key greater than or equal to the target. | 3 | 0 |
| `predecessor` | Return the greatest key strictly below the target. | 3 | 0 |
| `successor` | Return the smallest key strictly above the target. | 3 | 0 |
| `depth` | Return the target depth, counting the root as 0. | 3 | 0 |
| `level` | Return the target level, counting the root as 1. | 3 | 0 |
| `path` | Return the search path including the target when found. | 3 | 1 |
| `path_without_target` | Return the search path excluding the target. | 3 | 1 |
| `insert` | Insert one key, ignoring duplicates. | 2 | 0 |
| `insert_many` | Insert multiple keys, ignoring duplicates. | 4 | 2 |
| `range_count` | Count eligible keys within a range. | 5 | 5 |
| `range_sum` | Sum eligible keys or their transformed values within a range. | 7 | 13 |
| `range_values` | Return eligible keys within a range. | 6 | 9 |
| **Total** | | **56** | **31** |

Variants change empty or missing-result values (`None`, `-1`, or `0`), path order and missing-target behavior, insertion order and reporting, range endpoint inclusion, parity filters, output order, and sum transformations (identity, square, or absolute value).

Combination tasks apply multiple such conditions to one operation; for example, a range query may request only even keys in descending order.

All three self-trained models in the following table are evaluated after SFT, without RL.

| Model | Basic / single condition | Condition combinations | Overall |
| --- | ---: | ---: | ---: |
| Base + SFT | 28/56 | 3/31 | 31/87 (35.6%) |
| Base + Loop + SFT | 48/56 | 16/31 | 64/87 (73.6%) |
| Base + Loop + LoRA + SFT | 48/56 | 26/31 | 74/87 (85.1%) |
| Qwen2.5-Coder-0.5B Base | 6/56 | 0/31 | 6/87 (6.9%) |
| Qwen2.5-Coder-0.5B Instruct | 5/56 | 0/31 | 5/87 (5.7%) |

### Additional compositional evaluation dataset

We added 27 evaluation tasks that combine conditions of existing BST operations in ways absent from the SFT and RL training sets: 2 range-count, 9 range-values, and 16 range-sum tasks.

Each task combines three or four conditions and has 164 execution cases; we verified that no complete task specification or reference solution duplicates a training example, including after entry-point renaming.

This dataset is used only for evaluation, not for training rewards or checkpoint selection.

| Model | Range count | Range values | Range sum | Overall |
| --- | ---: | ---: | ---: | ---: |
| Base + SFT | 0/2 | 0/9 | 0/16 | 0/27 (0.0%) |
| Base + Loop + SFT | 0/2 | 0/9 | 1/16 | 1/27 (3.7%) |
| Base + Loop + LoRA + SFT | 0/2 | 3/9 | 7/16 | 10/27 (37.0%) |
| Qwen2.5-Coder-0.5B Base | 0/2 | 0/9 | 0/16 | 0/27 (0.0%) |
| Qwen2.5-Coder-0.5B Instruct | 0/2 | 0/9 | 0/16 | 0/27 (0.0%) |

Strict and interface-normalized scores agree for every model; all self-trained answers terminated normally, while three Qwen Base answers reached its 2,048-token limit.

The final model solves nine more tasks on the additional dataset than the paired Loop model.

## 7. Reinforcement Learning

We apply Group Relative Policy Optimization (GRPO) to the Base + Loop + LoRA model after SFT.

For each prompt, the model samples eight solutions, each receiving a reward of 1 if it passes all execution tests and 0 otherwise.

Rewards are normalized within each group to estimate advantages without a separate value model.

The objective combines a clipped policy loss with a KL penalty against the original SFT model.

Training uses 16 prompts per batch, two optimizer updates per batch, a learning rate of 0.000001, clipping of 0.2, and a KL coefficient of 0.02.

### Results

Each task receives one greedily decoded solution and counts as correct only if all its execution tests pass.

| Evaluation | After SFT | After RL |
| --- | ---: | ---: |
| Original: basic / single condition | 48/56 | 50/56 |
| Condition combinations | 26/31 | 26/31 |
| Original dataset overall | 74/87 (85.1%) | 78/87 (89.7%) |
| Additional compositional dataset | 10/27 (37.0%) | 12/27 (44.4%) |


