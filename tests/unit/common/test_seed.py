"""Test reproducibility helpers."""

from __future__ import annotations

import random

import numpy as np
import pytest

from helix.common.seed import set_global_seed


def test_set_global_seed_replays_python_and_numpy_rngs() -> None:
    set_global_seed(123)
    first_python = random.random()
    first_numpy = np.random.random()

    set_global_seed(123)

    assert random.random() == first_python
    assert np.random.random() == first_numpy


def test_set_global_seed_rejects_invalid_seed() -> None:
    with pytest.raises(ValueError):
        set_global_seed(-1)
