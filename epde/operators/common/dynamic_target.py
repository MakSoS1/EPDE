"""Research-only temporal anchoring during equation discovery (NIR1 H9).

An implicit EPDE equation can have practically zero residual without describing
dynamics: u_x * u_xx = u * u_xxx holds for any single spatial Fourier mode.
For evolution problems this is a valid spatial identity, NOT a recovered
governing time-evolution law.

H9 restricts the *explanatory target* of the right-part selector to a PURE
time derivative of its own state variable. This does not disclose the true
coefficient, RHS terms, derivative order, or true equation. The physical
assumption "we seek an evolution law in x0 (time)" IS an explicit additional
prior: controlled experiments must supply the same prior to the baseline,
or label the comparison as prior-assisted, not a new regularizer win.

Disabled by default. No change in population/epochs/token library/observations.
"""
from __future__ import annotations

from epde.evaluators import simple_function_evaluator


def is_temporal_target(term, variable) -> bool:
    """Accept only one *bare* derivative factor in temporal coordinate 0.

    A time derivative multiplied by x or u is not a valid implicit
    normalization target here. Products may still enter the RHS.
    """
    factors = getattr(term, "structure", ())
    if len(factors) != 1:
        return False
    factor = factors[0]
    code = getattr(factor, "deriv_code", None)
    return bool(
        getattr(factor, "variable", None) == variable
        and getattr(factor, "is_deriv", False)
        and code is not None
        and code != [None]
        and 0 in code
        and getattr(getattr(factor, "evaluator", None), "_evaluator", None)
        == simple_function_evaluator
        and term.contains_deriv(variable)
    )


def ensure_temporal_target(equation, *, max_attempts: int = 400):
    """Install at least one target without learning oracle ground truth.

    Randomly draw from the *existing* token pool. At the term cap, replace
    a non-target term; all complexity budgets and token families are held
    fixed. Replacing one term can sacrifice RHS diversity: document and
    measure this ablation against an equal-prior baseline.
    """
    from epde.structure.main_structures import Term

    if any(is_temporal_target(t, equation.main_var_to_explain)
           for t in equation.structure):
        return False
    allowed = int(equation.metaparameters["max_terms_number"]["value"])
    if allowed < 2:
        raise ValueError("Time-target experiment requires at least two terms")
    for _ in range(max_attempts):
        candidate = Term(equation.pool,
                         mandatory_family=equation.main_var_to_explain,
                         max_factors_in_term=1,
                         create_derivs=True)
        if not is_temporal_target(candidate, equation.main_var_to_explain):
            continue
        if any(t.factors_labels == candidate.factors_labels for t in equation.structure):
            continue
        if len(equation.structure) < allowed:
            equation.structure.append(candidate)
        else:
            # Keep native randomized EOS terms as much as possible: prefer
            # replacing an existing spatial derivative identity target.
            non_temporal = [
                i for i, term in enumerate(equation.structure)
                if not is_temporal_target(term, equation.main_var_to_explain)
            ]
            if not non_temporal:
                return False
            idx = next((i for i in non_temporal
                        if equation.structure[i].contains_deriv(
                            equation.main_var_to_explain)), non_temporal[-1])
            equation.structure[idx] = candidate
        equation._invalidate_label_cache()
        return True
    raise RuntimeError(
        "NIR1 temporal target could not be generated from the existing "
        "token pool; require axis-0 time derivatives for the explained variable")
