from __future__ import annotations

import os
import random
import tempfile
from typing import IO, BinaryIO

import numpy as np
import torch
from torch import nn


CheckpointDestination = str | os.PathLike[str] | BinaryIO | IO[bytes]


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: CheckpointDestination,
    *,
    config: dict | None = None,
    tokens_seen: int = 0,
) -> None:
    """Serialize all state required to resume training."""
    numpy_state = np.random.get_state()
    payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
            "config": config,
            "tokens_seen": tokens_seen,
            "torch_rng": torch.get_rng_state(),
            "python_rng": random.getstate(),
            "numpy_rng": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "mps_rng": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
    }
    if isinstance(out, (str, os.PathLike)):
        # Replace only after a complete save; interruption leaves the previous checkpoint intact.
        directory = os.path.dirname(os.path.abspath(out))
        os.makedirs(directory, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".checkpoint-", suffix=".pt", dir=directory)
        os.close(fd)
        try:
            torch.save(payload, temporary)
            os.replace(temporary, out)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    else:
        torch.save(payload, out)


def load_checkpoint(
    src: CheckpointDestination,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Restore training state and return the saved iteration."""
    checkpoint = torch.load(src, map_location="cpu", weights_only=True)
    if any("first_moment" in state for state in checkpoint["optimizer"]["state"].values()):
        raise ValueError("This checkpoint uses the old custom AdamW; start a fresh run with the new optimizer.")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if "torch_rng" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng"])
        random.setstate(checkpoint["python_rng"])
        name, keys, position, has_gauss, cached_gaussian = checkpoint["numpy_rng"]
        np.random.set_state((name, np.array(keys, dtype=np.uint32), position, has_gauss, cached_gaussian))
        if torch.cuda.is_available() and checkpoint["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        if torch.backends.mps.is_available() and checkpoint.get("mps_rng") is not None:
            torch.mps.set_rng_state(checkpoint["mps_rng"])
    return checkpoint["iteration"]
