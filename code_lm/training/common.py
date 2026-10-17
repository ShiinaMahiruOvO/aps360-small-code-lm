from __future__ import annotations

from contextlib import nullcontext
import torch
import numpy.typing as npt
import numpy as np
from torch import nn
from torch.nn import functional as F

from code_lm.data.pretrain import get_batch


def autocast_context(device: torch.device, amp_dtype: torch.dtype | None):
    return nullcontext() if amp_dtype is None else torch.autocast(device.type, dtype=amp_dtype)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataset: npt.NDArray[np.integer],
    batch_size: int,
    context_length: int,
    device: str | torch.device,
    num_batches: int,
    amp_dtype: torch.dtype | None = None,
    seed: int = 1234,
) -> float:
    """Estimate mean validation loss over freshly sampled batches."""
    if num_batches <= 0:
        raise ValueError(f"num_batches must be positive, got {num_batches}")

    was_training = model.training
    model.eval()
    total_loss = torch.zeros((), device=device)
    rng = np.random.default_rng(seed)
    try:
        for _ in range(num_batches):
            inputs, targets = get_batch(dataset, batch_size, context_length, device, rng=rng)
            with autocast_context(torch.device(device), amp_dtype):
                logits = model(inputs)
                total_loss += F.cross_entropy(logits.float().flatten(0, 1), targets.flatten())
    finally:
        model.train(was_training)
    return float(total_loss / num_batches)
    
