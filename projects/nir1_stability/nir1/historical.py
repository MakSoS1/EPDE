"""Auditable 1-D proxy for historical main-branch vclog+swap.

The archive used PDE spatial-axis spectral factors and EPDE's own weighted
constants-only variance. A generic ResearchDesign cannot reconstruct those
PDE-specific statistics. Never mark this proxy as a successful replay.
"""

from __future__ import annotations

import numpy as np

from .design import ResearchDesign
from .diagnostics import Diagnostics


def _spectral_inflation(residual: np.ndarray, groups: np.ndarray, max_lag: int = 8) -> float:
    factors = []
    for group in sorted(set(groups)):
        values = residual[groups == group]
        values = values - np.mean(values)
        scale = float(np.dot(values, values))
        if len(values) < 2 * max_lag or scale < 1e-25:
            continue
        ratios = [float(np.dot(values[lag:], values[:-lag]) / scale)
                  for lag in range(1, max_lag + 1)]
        factors.append(1. + 2. * sum((1. - lag / (max_lag + 1)) * corr
                                     for lag, corr in enumerate(ratios, 1)))
    return float(np.clip(np.median(factors), .01, 100.)) if factors else 1.


def historical_vclog_weights(design: ResearchDesign, diagnostics: Diagnostics) -> tuple[np.ndarray, dict[str, object]]:
    """Return vclog score and explicit archival-nonreproduction provenance."""
    factor = _spectral_inflation(design.y - design.X @ diagnostics.beta,
                                 design.environment_id)
    beta2 = diagnostics.beta**2
    denominator_floor = max(float(np.median(beta2[beta2 > 0])) * 1e-10
                            if np.any(beta2 > 0) else 0., 1e-18)
    var = np.maximum(diagnostics.expected_variance, 0.)
    values = np.log1p(factor * var / np.maximum(beta2, denominator_floor))
    meta = {
        "historical_status": "archival_unreproduced",
        "historical_reported_perfect": {"vclog_swap": "58/76", "production": "45/76"},
        "reason": "1D proxy; original constants-only variance and multi-axis spectral factors not reconstructed",
        "spectral_factor_1d_proxy": factor,
        "source": "main/projects/thesis/stability_audit.md",
    }
    return values, meta
