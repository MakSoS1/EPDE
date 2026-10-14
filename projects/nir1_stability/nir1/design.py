"""Validated, reproducible weighted candidate-regression inputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ResearchDesign:
    X: np.ndarray
    y: np.ndarray
    sample_weight: np.ndarray
    environment_id: np.ndarray
    token_names: tuple[str, ...]
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        X = np.asarray(self.X, dtype=np.float64)
        y = np.asarray(self.y, dtype=np.float64)
        w = np.asarray(self.sample_weight, dtype=np.float64)
        groups = np.asarray(self.environment_id, dtype=str)
        tokens = tuple(self.token_names)
        if X.ndim != 2 or not X.shape[0] or not X.shape[1]:
            raise ValueError("X must have shape (n > 0, p > 0)")
        n, p = X.shape
        if y.shape != (n,) or w.shape != (n,) or groups.shape != (n,):
            raise ValueError("y, sample_weight and environment_id shapes must match X rows")
        if len(tokens) != p or any(not t or not isinstance(t, str) for t in tokens) or len(set(tokens)) != p:
            raise ValueError("token_names must be distinct nonempty strings, one per X column")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("Design and target entries must be finite")
        if not np.isfinite(w).all() or np.any(w < 0) or np.sum(w) <= 0:
            raise ValueError("sample_weight must be finite, nonnegative, and positive in sum")
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "sample_weight", w)
        object.__setattr__(self, "environment_id", groups)
        object.__setattr__(self, "token_names", tokens)
        object.__setattr__(self, "metadata", dict(self.metadata))
