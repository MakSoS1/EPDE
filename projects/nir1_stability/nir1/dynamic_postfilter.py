"""H9 equal-prior control: require a temporal target only AFTER default EPDE.

This is the crucial ablation separating adding a physics prior during
evolution from simply applying the SAME prior at final Pareto selection.
No new EPDE evolution, terms, trajectories, or optimizer seeds are added.
"""
from __future__ import annotations

import re
from collections.abc import Sequence


_TEMPORAL_TARGET = re.compile(
    r"(?:[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?\s*\*\s*)?"
    r"d(?:\^\d+)?[A-Za-z][A-Za-z0-9_]*/dx0(?:\^\d+)?\{[^{}]+\}"
)


def is_explicit_temporal_equation(text: str) -> bool:
    if "=" not in text:
        return False
    right = text.split("=", 1)[1].strip()
    return _TEMPORAL_TARGET.fullmatch(right) is not None


def select_temporal_postfilter(front: Sequence[Sequence[str]],
                               objective_vectors: Sequence[Sequence[float]],
                               *, truth_labels=None):
    """Return global Pareto index or None; label is intentionally ignored."""
    del truth_labels
    if len(front) != len(objective_vectors):
        raise ValueError("Front and objectives must be paired")
    valid = [
        i for i, equations in enumerate(front)
        if equations and all(is_explicit_temporal_equation(s) for s in equations)
    ]
    if not valid:
        return None
    from projects.pic.epde_bench.metrics import select_compromise
    local = select_compromise([objective_vectors[i] for i in valid])
    return valid[local] if local is not None else None
