"""Truth-independent whole-equation ranking from coefficient diagnostics."""

from __future__ import annotations

import numpy as np

from .diagnostics import Diagnostics


def score_equation(diagnostics: Diagnostics, *, aggregator: str = "upper_quantile") -> float:
    """Lower is more stable; return +inf when evidence is unidentifiable.

    Never turn all-NaN/all-unreliable input into an artificially good zero.
    """
    if aggregator not in {"upper_quantile", "mean", "max"}:
        raise ValueError(f"Unsupported aggregator: {aggregator}")
    values = diagnostics.excess_variation[diagnostics.reliable]
    if values.size == 0 or not np.isfinite(values).all():
        return float("inf")
    if aggregator == "max":
        return float(np.max(values))
    if aggregator == "mean":
        return float(np.mean(values))
    return float(np.quantile(values, .9))
