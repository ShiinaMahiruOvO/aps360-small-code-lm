from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor


class ShuffledTokenBlocks:
    """A fixed permutation of non-overlapping T-token input/target blocks.

    The offset counts sequences, not tokens. There is no implicit epoch wrap;
    rebuilding this object with the same data/seed restores the exact order.
    """

    def __init__(self, dataset, context_length: int, seed: int = 2026):
        if context_length <= 0 or len(dataset) <= context_length:
            raise ValueError("Need positive context and at least context_length + 1 tokens")
        self.dataset = dataset
        self.context_length = context_length
        self.order = np.random.default_rng(seed).permutation((len(dataset) - 1) // context_length)

    def batch(self, offset: int, batch_size: int, device: str | torch.device):
        if offset < 0 or batch_size <= 0 or offset + batch_size > len(self.order):
            raise ValueError("Requested batch exceeds the single-pass block order")
        starts = self.order[offset:offset + batch_size] * self.context_length
        indices = starts[:, None] + np.arange(self.context_length + 1)[None, :]
        tokens = torch.from_numpy(np.asarray(self.dataset[indices], dtype=np.int64))
        device = torch.device(device)
        if device.type == "cuda":
            tokens = tokens.pin_memory()
        tokens = tokens.to(device, non_blocking=device.type == "cuda")
        return tokens[:, :-1], tokens[:, 1:]


def get_batch(
    dataset: npt.NDArray[np.integer],
    batch_size: int,
    context_length: int,
    device: str | torch.device,
    rng: np.random.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Sample language-model inputs and one-token-shifted targets."""
    
    if batch_size <= 0 or context_length <= 0 or len(dataset) <= context_length:
        raise ValueError("Need positive batch/context sizes and at least context_length + 1 tokens")
    randint = np.random.randint if rng is None else rng.integers
    starts = randint(
        0,
        len(dataset) - context_length,
        size=batch_size,
    )

    indices = starts[:, None] + np.arange(context_length + 1)[None, :]
    tokens = torch.from_numpy(np.asarray(dataset[indices], dtype=np.int64))
    device = torch.device(device)
    if device.type == "cuda":
        tokens = tokens.pin_memory()
    tokens = tokens.to(device=device, non_blocking=device.type == "cuda")
    return tokens[:, :-1], tokens[:, 1:]
