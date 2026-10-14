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


def select_sparsefront(front: Sequence[Sequence[str]],
                       objective_vectors: Sequence[Sequence[float]],
                       *, term_cost: float = 0.003,
                       high_derivative_cost: float = 0.001,
                       truth_labels=None) -> int | None:
    """Truth-free discrepancy/parsimonious-structure selection, NIR1 v2.

    S1 *development* study (NOT independent evidence) chose both fixed costs.
    This selector uses only the objective values and expression text. For
    coupled systems the first objective of each equation is averaged.
    Complexity counts nonzero RHS terms, while high derivative order costs
    max(0, derivative_order-2). The accepted S2 list MUST be frozen before
    running, otherwise post-selection gains are overfitted S1 results.
    """
    import re

    del truth_labels
    if not front:
        return None
    values = np.asarray(objective_vectors, dtype=float)
    if values.ndim != 2 or len(values) != len(front) or values.shape[1] % 2 != 0:
        raise ValueError("front and objective vectors must align in [discrepancy, instability] pairs")
    if not np.isfinite(values).all():
        valid = np.isfinite(values).all(axis=1)
    else:
        valid = np.ones(len(front), dtype=bool)
    if not valid.any():
        return None
    if term_cost < 0 or high_derivative_cost < 0:
        raise ValueError("Complexity and derivative costs must be nonnegative")

    scores = np.full(len(front), np.inf)
    for index, equations in enumerate(front):
        if not valid[index]:
            continue
        if not equations or len(equations) * 2 != values.shape[1]:
            raise ValueError("Each equation needs exactly two objectives")
        term_count = 0
        derivative_excess = 0
        for equation in equations:
            if "=" not in equation:
                raise ValueError("Unparseable discovered equation without '='")
            lhs = equation.split("=", 1)[0]
            for term in re.split(r"(?<![eE])\+", lhs):
                if not re.search(r"\{[^{}]*\}", term):
                    continue
                # The token presence check is exact for normal EPDE strings;
                # constants and numerically zero terms are not counted.
                scalar = 1.0
                for item in term.split("*"):
                    try:
                        scalar *= float(item.strip())
                    except ValueError:
                        continue
                if abs(scalar) < 1e-12:
                    continue
                term_count += 1
                for match in re.finditer(r"d(?:\^(\d+))?u/dx", term):
                    order = int(match.group(1)) if match.group(1) else 1
                    derivative_excess += max(0, order - 2)
        # Discrepancy is already a dimensionless relative-error objective.
        # No contrastive oracles and no hidden reweighting based on true terms.
        scores[index] = (float(np.mean(values[index, ::2])) +
                         term_cost * term_count +
                         high_derivative_cost * derivative_excess)
    return int(np.argmin(scores))
