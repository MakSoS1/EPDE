"""Candidate-level adaptive elastic net with checked convergence and refit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .design import ResearchDesign
from .diagnostics import Diagnostics
from .gram import GramBlocks, accumulate_gram


@dataclass(frozen=True)
class SparseFit:
    beta: np.ndarray
    support: np.ndarray
    objective: float
    converged: bool
    kkt_residual: float
    fallback_reason: str | None
    iterations: int


def _soft_threshold(value: float, threshold: float) -> float:
    return float(np.sign(value) * max(abs(value) - threshold, 0.))


def _standardized_gram(design: ResearchDesign,
                       moments: GramBlocks | None = None) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    moments = accumulate_gram(design) if moments is None else moments
    scale = np.sqrt(np.clip(np.diag(moments.G) / moments.weight_sum, 0., np.inf))
    safe_scale = np.where(scale > 0., scale, 1.)
    n = moments.weight_sum
    gram = moments.G / np.outer(safe_scale, safe_scale) / n
    rhs = moments.b / safe_scale / n
    return gram, rhs, moments.yy / n, safe_scale


def fit_weighted_en(design: ResearchDesign, *, l1: float, l2: float,
                    weights: np.ndarray, tol: float = 1e-8,
                    max_iter: int = 2000, warm_start: np.ndarray | None = None,
                    _precomputed_gram: GramBlocks | None = None) -> SparseFit:
    """Solve 1/(2Σw)||y-Xβ||²_W + λ₁ Σa_j|θ_j| + λ₂||θ||²/2.

    θ are standardized coefficients. The intercept token, if explicitly
    named ``__intercept__``, is unpenalized. Nonconvergence is surfaced.
    """
    if l1 < 0 or l2 < 0 or tol <= 0 or max_iter <= 0:
        raise ValueError("penalty, tol and max_iter must be nonnegative/positive")
    p = design.X.shape[1]
    penalties = np.asarray(weights, dtype=float)
    if penalties.shape != (p,) or not np.isfinite(penalties).all() or np.any(penalties <= 0):
        raise ValueError("penalty weights must be positive finite values for all columns")
    gram, b, yy, scales = _standardized_gram(design, _precomputed_gram)
    penalty_l1 = l1 * penalties.copy()
    penalty_l2 = np.full(p, l2)
    for index, name in enumerate(design.token_names):
        if name == "__intercept__":
            penalty_l1[index] = 0.
            penalty_l2[index] = 0.
    if warm_start is None:
        theta = np.zeros(p)
    else:
        ws = np.asarray(warm_start, dtype=float)
        if ws.shape != (p,) or not np.isfinite(ws).all():
            raise ValueError("warm_start must be finite and match column count")
        theta = ws * scales

    rank = int(np.linalg.matrix_rank(gram, tol=1e-11))
    reason = "rank_deficient" if rank < p else None
    converged = False
    violation = float("inf")
    for iteration in range(1, max_iter + 1):
        for j in range(p):
            denominator = gram[j, j] + penalty_l2[j]
            if denominator <= 0:
                theta[j] = 0.
                continue
            residual = b[j] - (gram[j, :] @ theta - gram[j, j] * theta[j])
            theta[j] = _soft_threshold(residual, penalty_l1[j]) / denominator
        grad = gram @ theta - b + penalty_l2 * theta
        active = np.abs(theta) > 1e-12
        kkt = np.maximum(np.abs(grad) - penalty_l1, 0.)
        kkt[active] = np.abs(grad[active] + penalty_l1[active] * np.sign(theta[active]))
        violation = float(np.max(kkt))
        if violation <= tol:
            converged = True
            break
    if not converged:
        reason = "; ".join(filter(None, [reason, "kkt_not_met"]))

    objective = (.5 * (yy - 2 * np.dot(b, theta) + theta @ gram @ theta)
                 + float(np.dot(penalty_l1, np.abs(theta)))
                 + .5 * float(np.dot(penalty_l2, theta * theta)))
    physical_beta = theta / scales
    return SparseFit(physical_beta, np.abs(theta) > 1e-10,
                     float(objective), converged, violation, reason, iteration)


def build_penalty_weights(diagnostics: Diagnostics, strategy: str, gamma: float,
                          bounds: tuple[float, float]) -> np.ndarray:
    """Bound all penalties; low Q never causes automatic hard deletion."""
    if not np.isfinite(gamma) or gamma < 0 or len(bounds) != 2 or not 0 < bounds[0] <= bounds[1]:
        raise ValueError("invalid penalty gamma or bounds")
    q = np.clip(diagnostics.identifiability, 0., 1.)
    excess = np.maximum(diagnostics.excess_variation, 0.)
    if strategy == "uniform":
        raw = np.ones_like(q)
    elif strategy == "instability":
        raw = 1. + gamma * np.tanh(excess)
    elif strategy in {"identifiability", "hybrid"}:
        raw = (1. + gamma * np.tanh(excess) * q) * (.25 + .75 * q)
    elif strategy == "adaptive_lasso":
        amplitude = np.abs(diagnostics.beta)
        reference = max(float(np.median(amplitude)), 1e-12)
        raw = (reference / np.maximum(amplitude, reference * .05)) ** gamma
    else:
        raise ValueError(f"Unknown penalty strategy: {strategy}")
    raw = np.nan_to_num(raw, nan=bounds[1], posinf=bounds[1], neginf=bounds[0])
    return np.clip(raw, *bounds)


def post_refit(design: ResearchDesign, support: np.ndarray, ridge: float = 0.) -> np.ndarray:
    """Return full-length post-selection coefficients; excluded terms are 0."""
    use = np.asarray(support, dtype=bool)
    p = design.X.shape[1]
    if use.shape != (p,) or ridge < 0:
        raise ValueError("Invalid support mask or ridge")
    beta = np.zeros(p)
    if not np.any(use):
        return beta
    X = design.X[:, use] * np.sqrt(design.sample_weight)[:, None]
    y = design.y * np.sqrt(design.sample_weight)
    if ridge:
        beta[use] = np.linalg.solve(X.T @ X + ridge * np.eye(X.shape[1]), X.T @ y)
    else:
        beta[use] = np.linalg.lstsq(X, y, rcond=None)[0]
    return beta
