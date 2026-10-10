"""Fixed-front selection using objective values only (never simulator truth)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def select_normalized_utopia(objective_vectors: Sequence[Sequence[float]],
                            *, truth_labels=None) -> int | None:
    """Argmin Euclidean distance to per-axis ideal under robust [0,1] scaling.

    Both axes are minimized. Constant axes contribute zero. Invalid rows are
    ineligible. Ties resolve by original row index and no label/oracle is read.
    ``truth_labels`` exists solely to demonstrate noninterference in tests.
    """
    del truth_labels
    if len(objective_vectors) == 0:
        return None
    values = np.asarray(objective_vectors, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("front must be an (n, objectives>=2) array")
    good = np.all(np.isfinite(values), axis=1)
    if not good.any():
        return None
    valid = values[good]
    minima = valid.min(axis=0)
    extent = valid.max(axis=0) - minima
    scale = np.where(extent > 0, extent, 1)
    distances = np.linalg.norm((valid - minima) / scale, axis=1)
    return int(np.flatnonzero(good)[np.argmin(distances)])
