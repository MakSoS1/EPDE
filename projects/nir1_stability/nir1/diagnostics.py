"""Coefficient instability split into model variation and noise uncertainty.

The uncalibrated E statistic is a *diagnostic*, not a 95% confidence bound
when X contains errors or the trajectory errors are temporally dependent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .design import ResearchDesign


@dataclass(frozen=True)
class Diagnostics:
    beta: np.ndarray
    excess_variation: np.ndarray
    expected_variance: np.ndarray
    identifiability: np.ndarray
    reliable: np.ndarray
    rank: int
    condition: float
    warnings: tuple[str, ...]


def estimate_diagnostics(design: ResearchDesign, *, ridge: float = 1e-8,
                         rtol: float = 1e-10) -> Diagnostics:
    """SVD-stable weighted OLS and per-environment excess coefficient spread.

    We work in globally standardized feature coordinates, so Q and E are
    unchanged by any positive per-column unit transformation. `ridge` is
    currently a documented safety parameter, not silently added to OLS.
    """
    if ridge < 0 or rtol <= 0:
        raise ValueError("ridge must be nonnegative and rtol positive")
    X, y, weights = design.X, design.y, design.sample_weight
    root_w = np.sqrt(weights)
    scales = np.linalg.norm(X * root_w[:, None], axis=0)
    safe_scales = np.where(scales > 0, scales, 1.)
    standardized = X / safe_scales
    A = standardized * root_w[:, None]
    target = y * root_w
    p = X.shape[1]
    singular = np.linalg.svd(A, compute_uv=False)
    threshold = rtol * singular[0] if len(singular) else 0.
    rank = int(np.sum(singular > threshold))
    condition = (float(singular[0] / singular[-1])
                 if len(singular) and singular[-1] > threshold else float("inf"))
    beta_scaled = np.linalg.lstsq(A, target, rcond=rtol)[0]
    beta = beta_scaled / safe_scales

    q = np.empty(p)
    for j in range(p):
        if scales[j] == 0:
            q[j] = 0.
            continue
        neighbors = np.delete(A, j, axis=1)
        if neighbors.shape[1] == 0:
            q[j] = 1.
            continue
        projected = neighbors @ np.linalg.lstsq(neighbors, A[:, j], rcond=rtol)[0]
        q[j] = float(np.clip(np.linalg.norm(A[:, j] - projected)**2 /
                             (np.linalg.norm(A[:, j])**2 + 1e-300), 0., 1.))

    coefficients = []
    uncertainty = []
    warnings: list[str] = []
    for group in sorted(set(design.environment_id)):
        mask = (design.environment_id == group) & (weights > 0)
        local_A, local_target = A[mask], target[mask]
        if local_A.shape[0] <= p or np.linalg.matrix_rank(local_A, tol=rtol) < p:
            warnings.append("insufficient_environment_samples")
            continue
        local_beta = np.linalg.lstsq(local_A, local_target, rcond=rtol)[0]
        coefficients.append(local_beta)
        rss = np.linalg.norm(local_target - local_A @ local_beta)**2
        sigma2 = rss / max(local_A.shape[0] - p, 1)
        local_cov = sigma2 * np.linalg.pinv(local_A.T @ local_A, rcond=rtol)
        uncertainty.append(np.clip(np.diag(local_cov), 0., np.inf))

    if len(coefficients) >= 2:
        D = np.var(np.asarray(coefficients), axis=0, ddof=1)
        U = np.mean(np.asarray(uncertainty), axis=0)
    else:
        D = np.zeros(p)
        U = np.full(p, np.inf)
        warnings.append("insufficient_environments")

    floor = max(float(np.median(np.abs(beta_scaled))) * .05,
                float(np.linalg.norm(target)) * 1e-10, 1e-12)
    E = np.maximum(D - U, 0.) / (beta_scaled**2 + floor**2)
    if rank < p:
        warnings.append("rank_deficient")
    if design.metadata.get("noise_X", 0) or design.metadata.get("errors_in_variables"):
        warnings.append("errors_in_variables")
    if design.metadata.get("model_class") == "variable":
        warnings.append("variable_coefficients")
    if condition > 1e8:
        warnings.append("ill_conditioned")
    reliable = ((q > 1e-5) & (rank == p) & np.isfinite(U)
                & ("errors_in_variables" not in warnings)
                & ("insufficient_environments" not in warnings))
    variance_phys = U / safe_scales**2
    return Diagnostics(beta, E, variance_phys, q, reliable, rank, condition,
                       tuple(dict.fromkeys(warnings)))
