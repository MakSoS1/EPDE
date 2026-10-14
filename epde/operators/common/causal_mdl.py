"""NIR1 H10: data-budget-aware minimum dynamic description length.

Temporal H9 prevents spatial-only identities, but finite-difference noise
can reward unnecessary high spatial derivatives (u_xxx, u_xxxx) that mimic
truncation errors. H10 adds an explicit predeclared structural cost to the
fitness being EVOLVED. It does not alter VWSR, tokens, points, epochs or
population size and never sees simulator truth.

Unlike fixed hand-tuned lambda, the cost unit is BIC's log(N_eff)/N_eff,
with N_eff the number of distinct temporal levels, not the correlated
number of spatial nodes. This is a heuristic effective-sample assumption,
not a claim that time points are iid. Assess robustness to time-correlation
and hidden physics independently.
"""
from __future__ import annotations

import math

import numpy as np


def temporal_effective_levels(data_shape) -> int:
    """Conservative time levels common to all trajectories (axis 0 is time)."""
    if not data_shape:
        raise ValueError("Temporal MDL requires explicit grid geometry")
    items = data_shape.values() if isinstance(data_shape, dict) else [data_shape]
    dims = []
    for shape in items:
        if shape is None or not len(shape):
            raise ValueError("Temporal MDL requires nonempty time geometry")
        levels = int(shape[0])
        if levels < 4:
            raise ValueError("At least 4 temporal levels needed for H10")
        dims.append(levels)
    return min(dims)


def dynamical_description_length(equation, data_shape) -> float:
    """Non-oracle complexity cost for ACTIVE RHS terms, without target.

    Penalize each additional RHS term (one unit) and each excess derivative
    order beyond the second (one per order), a common minimal-evolution
    inductive bias. Intercept enters as one ordinary constant term when
    nonzero. No learned labels, true structure or fitted coefficients are
    accessed except the sparsity-produced active mask.
    """
    n = temporal_effective_levels(data_shape)
    unit = math.log(n) / n
    tgt = equation.target_idx
    wi = np.asarray(equation.weights_internal, dtype=float)
    if tgt is None or len(wi) != len(equation.structure):
        raise ValueError("Model support not aligned with equation structure")
    complexity = int(abs(wi[-1]) > 1e-12)
    for idx, term in enumerate(equation.structure):
        if idx == tgt:
            continue
        k = equation.weight_index(idx, tgt)
        if abs(wi[k]) <= 1e-12:
            continue
        complexity += 1
        for factor in term.structure:
            code = getattr(factor,"deriv_code",None)
            if code is not None and code != [None]:
                # Spatial or temporal high orders both increase the
                # model's sensitivity to finite-difference uncertainty.
                complexity += max(0, len(code)-2)
    return float(unit * complexity)
