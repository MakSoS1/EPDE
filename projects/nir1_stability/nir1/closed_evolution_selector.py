"""H14: causal closure in the FINAL Pareto decision, no additional searches.

For an explicit scalar evolution law, its normalized temporal derivative
(target) must be algebraically isolated: RHS after rearrangement may only
contain state, space/time-independent coordinates, and SPATIAL derivatives.
An extra temporal derivative anywhere on the candidate RHS makes the law
an implicit/higher-differential-order consequence, not an explicit closure
in this hypothesis class. This is a strong PHYSICS PRIOR, not a universal
validity test. Damped waves u_tt = c²u_xx - gamma u_t genuinely violate
the assumed closure; challenge them in the prospective heldout suite.

Works only on EPDE's serialized single/multi-equation syntax, not a
symbolic general-purpose equivalence solver. Multiple equation systems
pass only if every component is in the valid canonical form.
"""
from __future__ import annotations

import re
from collections.abc import Sequence

from .causal_mdl_selector import select_min_causal_description

_TIME_FACTOR=re.compile(r"d(?:\^\d+)?[A-Za-z][A-Za-z0-9_]*/dx0(?:\^\d+)?\{[^{}]*\}")
_TARGET=re.compile(
    r"\s*(?:[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?\s*\*\s*)?"
    r"d(?:\^\d+)?[A-Za-z][A-Za-z0-9_]*/dx0(?:\^\d+)?\{[^{}]*\}\s*"
)


def is_closed_evolution_equation(equation: str) -> bool:
    """Full target is ONE bare temporal derivative; LHS has none."""
    if not isinstance(equation,str) or equation.count("=")!=1:
        return False
    lhs,target=(x.strip() for x in equation.split("=",1))
    return bool(_TARGET.fullmatch(target) and not _TIME_FACTOR.search(lhs))


def choose_causal_closed_front(front: Sequence[Sequence[str]],
                               objective_vectors: Sequence[Sequence[float]],
                               *, truth_labels=None) -> int | None:
    """Minimum evolved first fitness among closed candidate equations.

    No truth used, nonclosed candidates excluded. On complete lack of
    candidates return None (abstention; counts failure in accuracy), never
    silently reward an inadmissible candidate.
    """
    del truth_labels
    if len(front)!=len(objective_vectors):
        raise ValueError("Pareto data must align with candidate structures")
    valid=[i for i,eqs in enumerate(front)
           if eqs and all(is_closed_evolution_equation(eq) for eq in eqs)]
    if not valid:
        return None
    index=select_min_causal_description([objective_vectors[i] for i in valid])
    return valid[index] if index is not None else None
