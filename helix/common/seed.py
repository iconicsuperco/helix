"""Reproducibility helpers."""

from __future__ import annotations

import random
from typing import Final

import numpy as np
import torch

MAX_NUMPY_SEED: Final[int] = 2**32 - 1


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch global RNGs."""

    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("Seed must be an integer")
    if not 0 <= seed <= MAX_NUMPY_SEED:
        raise ValueError(f"Seed must be between 0 and {MAX_NUMPY_SEED}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
