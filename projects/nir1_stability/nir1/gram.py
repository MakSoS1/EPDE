"""Dense weighted sufficient statistics: exact subset reuse, O(np²) work."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .design import ResearchDesign


@dataclass(frozen=True)
class GramBlocks:
    G: np.ndarray
    b: np.ndarray
    yy: float
    weight_sum: float
    by_environment: dict[str, tuple[np.ndarray, np.ndarray, float]]


def accumulate_gram(design: ResearchDesign, chunk_rows: int = 8192) -> GramBlocks:
    """Accumulate by fixed-order chunks without storing weighted copies of X."""
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    X, y, w = design.X, design.y, design.sample_weight
    p = X.shape[1]
    gram = np.zeros((p, p), dtype=np.float64)
    target = np.zeros(p, dtype=np.float64)
    yy = 0.
    weight_sum = 0.
    env = {name: [np.zeros((p, p)), np.zeros(p), 0.]
           for name in sorted(set(design.environment_id.tolist()))}
    for start in range(0, len(y), chunk_rows):
        slc = slice(start, start + chunk_rows)
        xx, yy_chunk, ww = X[slc], y[slc], w[slc]
        gram += xx.T @ (ww[:, None] * xx)
        target += xx.T @ (ww * yy_chunk)
        yy += float(np.dot(ww, yy_chunk * yy_chunk))
        weight_sum += float(ww.sum())
        chunk_env = design.environment_id[slc]
        for name in sorted(set(chunk_env.tolist())):
            mask = chunk_env == name
            x_env, y_env, w_env = xx[mask], yy_chunk[mask], ww[mask]
            env[name][0] += x_env.T @ (w_env[:, None] * x_env)
            env[name][1] += x_env.T @ (w_env * y_env)
            env[name][2] += float(np.dot(w_env, y_env * y_env))
    return GramBlocks(gram, target, yy, weight_sum,
                      {name: (value[0], value[1], value[2]) for name, value in env.items()})


def subset_gram(gram: GramBlocks, indices: Sequence[int]) -> GramBlocks:
    idx = np.asarray(indices, dtype=int)
    if idx.ndim != 1 or np.any(idx < 0) or np.any(idx >= len(gram.b)) or len(set(idx.tolist())) != len(idx):
        raise ValueError("indices must be a unique, valid, 1D support subset")
    def subset(g: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return g[np.ix_(idx, idx)].copy(), b[idx].copy()
    g, b = subset(gram.G, gram.b)
    env = {name: (*subset(values[0], values[1]), values[2])
           for name, values in gram.by_environment.items()}
    return GramBlocks(g, b, gram.yy, gram.weight_sum, env)
