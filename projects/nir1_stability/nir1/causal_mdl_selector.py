"""H11: predeclare decision rule CONSISTENT with the H10 search objective.

The H10 search evolves a first objective equal to normalized WAPE +
BIC-like active structure penalty, and a second instability objective.
Selecting the utopia compromise can promote extra terms solely because
the instability axis happens to be exactly zero for overfit candidates.
Here final model = argmin(first/evolved MDL objective) among valid Pareto
entries. This is a simple data-only argmin, no oracle, no tunable weights.

Rule formulated AFTER inspecting the H10 eight-run development outputs.
The observed 8/8 is DEVELOPMENT ONLY. Must validate on fully disjoint
physical systems / equations before any confirmation claim.
"""
from __future__ import annotations

import math
from collections.abc import Sequence


def select_min_causal_description(objective_vectors: Sequence[Sequence[float]],
                                  *, truth_labels=None):
    """Return original front index, stable ties in recorded front order."""
    del truth_labels
    winner=None
    for idx, vec in enumerate(objective_vectors):
        if vec is None or not isinstance(vec,(list,tuple)) or not vec:
            continue
        try:
            v=[float(item) for item in vec]
        except (TypeError,ValueError,OverflowError):
            continue
        if not all(math.isfinite(x) for x in v):
            continue
        if winner is None or v[0] < winner[1]:
            winner=(idx,v[0])
    return winner[0] if winner else None
