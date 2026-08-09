"""Geometry boundaries for future bowling-ball modeling."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def as_vector3(value: ArrayLike) -> NDArray[np.float64]:
    """Convert an array-like value to a finite NumPy vector with three entries."""
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError("expected a vector with shape (3,)")
    if not np.all(np.isfinite(vector)):
        raise ValueError("vector components must be finite")
    return vector
