from __future__ import annotations

import math

def cosine_learning_rate_schedule(
    iteration: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Return the warmup-plus-cosine learning rate for an iteration."""
    if iteration < warmup_iters:
        return max_learning_rate * iteration / warmup_iters

    if iteration <= cosine_cycle_iters:
        progress = (
            (iteration - warmup_iters)
            / (cosine_cycle_iters - warmup_iters)
        )
        cosine_weight = 0.5 * (1 + math.cos(math.pi * progress))
        return min_learning_rate + cosine_weight * (
            max_learning_rate - min_learning_rate
        )

    return min_learning_rate
