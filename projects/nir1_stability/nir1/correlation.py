"""Truth-free correlated-term diagnosis and post-fit support swap."""

from __future__ import annotations

import numpy as np

from .design import ResearchDesign
from .regularizers import SparseFit, post_refit


def correlation_groups(design: ResearchDesign, threshold: float = .9) -> list[tuple[int, ...]]:
    """Connected components of weighted absolute correlation above threshold."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    X = design.X * np.sqrt(design.sample_weight)[:, None]
    norms = np.linalg.norm(X, axis=0)
    normed = X / np.maximum(norms, 1e-300)
    similarity = np.abs(normed.T @ normed)
    p = len(norms)
    parents = list(range(p))

    def root(i: int) -> int:
        while parents[i] != i:
            i = parents[i]
        return i

    for i in range(p):
        for j in range(i + 1, p):
            if similarity[i, j] >= threshold:
                parents[root(j)] = root(i)
    result: dict[int, list[int]] = {}
    for i in range(p):
        result.setdefault(root(i), []).append(i)
    return sorted((tuple(members) for members in result.values()), key=lambda group: group[0])


def _complexity(token: str) -> tuple[int, int]:
    return token.count("*") + 1, len(token)


def swap_refine(design: ResearchDesign, fit: SparseFit, *, tie_rel: float = .01) -> SparseFit:
    """Swap highly correlated survivor/removed terms only on measured RSS gain.

    A <=1% residual tie may prefer a *predeclared* simpler token; no access
    to true coefficients or benchmark labels is permitted.
    """
    if not 0 <= tie_rel < 1:
        raise ValueError("tie_rel must be within [0,1)")
    support = np.asarray(fit.support, dtype=bool).copy()
    if support.shape != (design.X.shape[1],):
        raise ValueError("invalid initial support shape")
    X, y, w = design.X, design.y, design.sample_weight
    norms = np.sqrt(np.sum(w[:, None] * X**2, axis=0))
    correlations = np.abs((X.T @ (w[:, None] * X)) / np.maximum(np.outer(norms, norms), 1e-300))

    def score(mask: np.ndarray) -> tuple[np.ndarray, float]:
        coeff = post_refit(design, mask)
        rss = float(np.dot(w, (y - X @ coeff)**2) / w.sum())
        return coeff, rss

    best_beta, best_rss = score(support)
    updated = False
    for j in range(len(support)):
        if not support[j]:
            continue
        for k in range(len(support)):
            if support[k] or k == j or correlations[j, k] < .9:
                continue
            trial = support.copy()
            trial[j], trial[k] = False, True
            beta_try, rss_try = score(trial)
            clear_win = rss_try < best_rss * (1. - 1e-9)
            tie_simpler = (rss_try <= best_rss * (1. + tie_rel)
                           and _complexity(design.token_names[k]) < _complexity(design.token_names[j]))
            if clear_win or tie_simpler:
                support, best_beta, best_rss = trial, beta_try, rss_try
                updated = True
                break
    reason = "post_fit_swap" if updated else fit.fallback_reason
    return SparseFit(best_beta if updated else fit.beta, support,
                     .5 * best_rss if updated else fit.objective,
                     fit.converged, fit.kkt_residual, reason, fit.iterations)
