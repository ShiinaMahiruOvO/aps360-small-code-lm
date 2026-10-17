"""Turn a question and its reference answer into a supervised training example."""

import json
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from code_lm.tokenizer import Tokenizer

IGNORE_INDEX = -100


class SFTDataset(Dataset):
    """One JSONL row is one example; only the answer and EOS contribute to loss.

    Required fields are ``prompt`` and ``answer``. The prompt already includes
    the task description, function signature and the final ``Solution:`` cue.
    """

    def __init__(self, path: Path, tokenizer: Tokenizer, context_length: int):
        self.examples = []
        self.group_ids = set()
        self.eos_id = tokenizer.token_to_id[b"<|endoftext|>"]

        with path.open() as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                prompt, answer = row["prompt"], row["answer"]
                if not isinstance(prompt, str) or not isinstance(answer, str) or not prompt or not answer:
                    raise ValueError(f"{path}:{line_number}: prompt and answer must be nonempty strings")
                if "<|endoftext|>" in prompt or "<|endoftext|>" in answer:
                    raise ValueError(f"{path}:{line_number}: EOS is added by the dataset, not the text")

                prompt_ids = tokenizer.encode(prompt)
                answer_ids = tokenizer.encode(answer) + [self.eos_id]
                # Our cleaning script caches IDs. Check them rather than silently
                # training with a different tokenizer that happens to have the same size.
                for field, expected in (("prompt_token_ids", prompt_ids), ("answer_token_ids", answer_ids)):
                    if field in row and row[field] != expected:
                        raise ValueError(f"{path}:{line_number}: {field} does not match this tokenizer")

                tokens = torch.tensor(prompt_ids + answer_ids, dtype=torch.long)
                if len(tokens) - 1 > context_length:
                    raise ValueError(f"{path}:{line_number}: example exceeds context length {context_length}")
                inputs = tokens[:-1]
                labels = tokens[1:].clone()
                # labels[j] predicts tokens[j + 1], so the first answer target
                # is at len(prompt_ids) - 1, not len(prompt_ids).
                labels[:len(prompt_ids) - 1] = IGNORE_INDEX
                self.examples.append((inputs, labels))
                if row.get("group_id") is not None:
                    self.group_ids.add(row["group_id"])

        if not self.examples:
            raise ValueError(f"No examples in {path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def collate_examples(examples, pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad each batch to its longest example, ignoring padding in the loss.

    With causal attention, real tokens cannot see padding to their right. We can
    therefore keep the existing Transformer unchanged and use EOS as the pad ID.
    A real EOS label is still supervised; padding labels are -100.
    """
    inputs, labels = zip(*examples, strict=True)
    return (
        pad_sequence(inputs, batch_first=True, padding_value=pad_id),
        pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX),
    )
