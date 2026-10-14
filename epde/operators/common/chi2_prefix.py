"""Opt-in exact-in-form cached cumulative-score statistic for RFE.

The original chi2_scores refits on each support, creates a per-observation
score and reduces it along grid axes, repeating n-sized work at every RFE
step. For a fixed full feature library A and fixed sample weights:

  marginal_score_d(t; S) =
       SUM_{i at level t} w_i A_ij y_i
     - SUM_{k in S} beta_k(S) SUM_{i at level t} w_i A_ij A_ik.

Therefore all per-axis increments can be recomputed from ONE cached tensor
of sufficient statistics. All subsequent support evaluations are small
matrix solves and tensor contractions, no n-sized matrix scan.

This cache is only enabled by NIR1_CHI2_PREFIX_CACHE=1 and only on
multidimensional grids >=5000 samples with bounded allocation. The classic
chi2_scores path remains the comparison oracle. Small/singular designs
fall back unchanged. The representation is exact over reals; floating-point
reassociation can differ, so end-to-end parity is mandatory before
claiming accuracy-preserving speed-up.
"""

from __future__ import annotations

import numpy as np


def prepare_chi2_prefix(features, target, weights, grid_shape,
                        full_gram, full_gy, *,
                        max_tensor_entries: int = 2_000_000,
                        min_samples: int = 5000):
    """Return sufficient statistics or None if memory/geometry is unsuitable."""
    X = np.asarray(features, dtype=float)
    y = np.asarray(target, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    n, p = X.shape
    shape = tuple(map(int, (() if grid_shape is None else grid_shape)))
    k = p + 1  # one appended intercept, identical to EPDE Gram convention
    if (len(shape) < 2 or int(np.prod(shape)) != n or n < min_samples or
            X.shape != (n, p) or y.shape != (n,) or w.shape != (n,) or
            not np.isfinite(w).all() or np.any(w < 0)):
        return None
    axis_levels = [shape[d] for d in range(len(shape)) if shape[d] >= 3]
    if not axis_levels or sum(axis_levels) * k*k > max_tensor_entries:
        return None
    if np.asarray(full_gram).shape != (k, k) or np.asarray(full_gy).shape != (k,):
        raise ValueError("Full Gram geometry does not match full feature library")
    A = np.column_stack((X, np.ones(n)))
    fourth = np.sum((w[:, None] * (A ** 2)) ** 2, axis=0)
    paths = []
    for d, nlevels in enumerate(shape):
        if nlevels < 3:
            continue
        moved_A = np.moveaxis(A.reshape(shape + (k,)), d, 0).reshape(nlevels, -1, k)
        moved_w = np.moveaxis(w.reshape(shape), d, 0).reshape(nlevels, -1)
        moved_y = np.moveaxis(y.reshape(shape), d, 0).reshape(nlevels, -1)
        marginal_xy = np.einsum('nm,nmj->nj', moved_w * moved_y, moved_A,
                                optimize=True)
        marginal_xx = np.einsum('nm,nmi,nmj->nij', moved_w, moved_A, moved_A,
                                optimize=True)
        paths.append((marginal_xy, marginal_xx, int(nlevels)))
    return {"gram": np.asarray(full_gram).copy(), "gy": np.asarray(full_gy).copy(),
            "fourth": fourth, "paths": paths, "shape": shape,
            "n_samples": n, "n_features": p}


def scores_from_chi2_prefix(cache, active_mask, *, ill_conditioned_cutoff=1e8):
    """Score the active sub-library, returning None to request classical fallback."""
    mask = np.asarray(active_mask, dtype=bool)
    if mask.shape != (cache["n_features"] + 1,):
        raise ValueError("Support mask not aligned with cached full library")
    ix = np.flatnonzero(mask)
    if not len(ix):
        return None
    gram = cache["gram"][np.ix_(ix, ix)]
    target = cache["gy"][ix]
    try:
        condition = np.linalg.cond(gram)
        if not np.isfinite(condition) or condition > ill_conditioned_cutoff:
            return None
        beta = np.linalg.solve(gram, target)
    except np.linalg.LinAlgError:
        return None
    with np.errstate(invalid='ignore', over='ignore'):
        denominator = (beta ** 2) * cache["fourth"][ix]
    if not np.isfinite(denominator).all():
        return None
    by_axis = []
    for xy, xx, levels in cache["paths"]:
        # Select precisely the model columns (including real intercept only
        # when selected); cached dummy intercept is not silently refit.
        small_xx = xx[:, ix][:, :, ix]
        increment = xy[:, ix] - np.einsum('tij,j->ti', small_xx, beta,
                                         optimize=True)
        cumulative = np.cumsum(increment, axis=0)
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            by_axis.append(np.sum(cumulative*cumulative, axis=0)
                           / (levels * denominator))
    return np.nan_to_num(np.mean(by_axis, axis=0))
