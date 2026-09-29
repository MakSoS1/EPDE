"""A candidate equation's residual, described independently of any solver.

Mirrors the DeepXDE backend exactly: ``DeepXDEAdapter._equation_system_to_pde_func``
(which terms, which coefficient, the intercept, the target and its sign),
``_factor_params`` (parameters read BY NAME) and ``_factor_value_with_map``
(dispatch on the factor's FAMILY, ``ftype``). The result is a plain
``EquationSpec`` that any backend evaluates on its own points:

    residual = sum_k coeff_k * prod(factors_k) + intercept - prod(target)

Factor kinds:

* ``grid``  -- ``coord[axis] ** power``
* ``trig``  -- ``sin|cos(freq * coord[axis]) ** power``
* ``const`` -- ``value ** power``
* ``var``   -- ``d^orders u_var ** power`` (``orders`` counts derivatives per
  EPDE axis, 0 = time). A plain variable token carries ``is_deriv=True`` with
  ``deriv_code=[None]``; only a real axis makes a derivative (the DXI bug that
  once dropped ``u^3``'s power).

Families the residual cannot express raise ``NotImplementedError``, as in DXI:
evaluating them as ``u`` or 1.0 silently was a bug there before.

``term_mass`` gives the pointwise cancellation denominator of
``Discrepancy._compute_scale_invariant``: |target| + sum_k |coeff_k term_k| +
|intercept|. The scale-invariant physics loss divides the residual by it.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class FactorSpec:
    kind: str                       #: 'grid' | 'trig' | 'const' | 'var'
    power: float = 1.0
    axis: Optional[int] = None      #: EPDE axis (grid / trig)
    func: Optional[str] = None      #: 'sin' | 'cos'
    freq: Optional[float] = None
    value: Optional[float] = None
    var: Optional[int] = None       #: index into the system's variables
    orders: Tuple[int, ...] = ()    #: derivatives per EPDE axis ('var')


@dataclass(frozen=True)
class TermSpec:
    coeff: float
    factors: Tuple[FactorSpec, ...]


@dataclass(frozen=True)
class EquationSpec:
    var: int                        #: the equation's main variable
    terms: Tuple[TermSpec, ...]
    intercept: float
    target: Tuple[FactorSpec, ...]


def factor_params(factor) -> dict:
    """The factor's parameters BY NAME, from ``params_description``
    (verbatim ``DeepXDEAdapter._factor_params``)."""
    description = getattr(factor, "params_description", None) or {}
    params = getattr(factor, "params", None)
    if params is None:
        return {}
    return {info["name"]: float(params[idx]) for idx, info in description.items()
            if isinstance(info, dict) and "name" in info}


def factor_spec(factor, var_idx_map: Dict[str, int], ndim: int) -> FactorSpec:
    """One factor as a spec; the dispatch of ``_factor_value_with_map``."""
    params = factor_params(factor)
    power = params.get("power", 1.0)
    family = str(getattr(factor, "ftype", ""))
    label = str(getattr(factor, "label", ""))
    variable = getattr(factor, "variable", None)

    if family == "grids":
        return FactorSpec("grid", power, axis=int(round(float(params["dim"]))))
    if family == "trigonometric":
        if label not in ("sin", "cos"):
            raise NotImplementedError(f"residual: unknown trigonometric token {label!r}")
        return FactorSpec("trig", power, axis=int(round(float(params["dim"]))),
                          func=label, freq=float(params["freq"]))
    if family == "constants":
        return FactorSpec("const", power, value=float(params["value"]))
    if variable not in var_idx_map:
        raise NotImplementedError(
            f"residual cannot express factor {label!r} of family {family!r} "
            f"(variable {variable!r}; system variables {sorted(var_idx_map)})")
    orders = [0] * ndim
    axes = [ax for ax in (getattr(factor, "deriv_code", None) or []) if ax is not None]
    if getattr(factor, "is_deriv", False) and axes:
        for ax in axes:
            orders[int(round(float(ax)))] += 1
    return FactorSpec("var", power, var=var_idx_map[variable], orders=tuple(orders))


def equation_spec(eq, var_idx_map: Dict[str, int], ndim: int) -> EquationSpec:
    """The residual of one fitted equation, by DXI's rules (803-841)."""
    use_weights = getattr(eq, "weights_final_evald", False) and hasattr(eq, "weights_final")
    tgt = eq.target_idx
    terms = []
    for term_idx, term in enumerate(eq.structure):
        if term_idx == tgt:
            continue
        # ``weight_index``: both weight vectors SKIP the target
        coeff = (float(eq.weights_final[eq.weight_index(term_idx, tgt)])
                 if use_weights else 1.0)
        terms.append(TermSpec(coeff, tuple(factor_spec(f, var_idx_map, ndim)
                                           for f in term.structure)))
    # the free coefficient only when the equation HAS one (support slot
    # ``weights_internal[-1]`` non-zero), as the DeepXDE residual
    has_free = use_weights and float(eq.weights_internal[-1]) != 0.0
    intercept = float(eq.weights_final[-1]) if has_free else 0.0
    target = tuple(factor_spec(f, var_idx_map, ndim) for f in eq.target.structure)
    return EquationSpec(var=var_idx_map[eq.main_var_to_explain], terms=tuple(terms),
                        intercept=intercept, target=target)


def system_specs(equation_or_system, ndim: int):
    """``(eq_list, var_names, specs)`` for an ``Equation`` or a ``SoEq``, in
    the order DXI uses (``vars_to_describe``)."""
    from epde.structure.main_structures import Equation, SoEq
    if isinstance(equation_or_system, Equation):
        eq_list = [equation_or_system]
        var_names = [equation_or_system.main_var_to_explain]
    elif isinstance(equation_or_system, SoEq):
        var_names = list(equation_or_system.vars_to_describe)
        eq_list = [equation_or_system.vals[v] for v in var_names]
    else:
        raise TypeError("Unsupported equation type")
    var_idx_map = {name: i for i, name in enumerate(var_names)}
    return eq_list, var_names, [equation_spec(eq, var_idx_map, ndim) for eq in eq_list]


def required_fields(specs: Sequence[EquationSpec]) -> List[Tuple[int, Tuple[int, ...]]]:
    """Every ``(var, orders)`` a residual reads, sorted (deterministic)."""
    need = set()
    for spec in specs:
        for fs in [f for t in spec.terms for f in t.factors] + list(spec.target):
            if fs.kind == "var":
                need.add((fs.var, fs.orders))
    return sorted(need)


def max_orders(specs: Sequence[EquationSpec], ndim: int) -> Tuple[int, ...]:
    out = [0] * ndim
    for _, orders in required_fields(specs):
        out = [max(a, b) for a, b in zip(out, orders)]
    return tuple(out)


# ------------------------------------------------------------ evaluation
def cancellation_ratio(r, mass, backend):
    """``r / mass`` with the one exact rule the ratio needs and no epsilon.

    ``|r| <= mass`` by the triangle inequality, so the ratio lies in [0, 1].
    The only singular point is ``mass == 0``, where every term vanishes; the
    equation then holds trivially and the ratio is 0. The denominator is
    replaced by 1 there BEFORE dividing (a "double where"): dividing by 0 and
    masking afterwards would still put a NaN into the gradient.
    """
    if not hasattr(mass, "shape"):
        return r / mass if mass > 0 else 0.0 * r
    positive = mass > 0
    safe = backend.where(positive, mass, backend.ones_like(mass))
    return backend.where(positive, r / safe, backend.zeros_like(r))


def inverse_mass(mass: np.ndarray) -> np.ndarray:
    """1 / mass, and 0 where the mass is exactly 0 (no information there)."""
    mass = np.asarray(mass, dtype=np.float64)
    out = np.zeros_like(mass)
    np.divide(1.0, mass, out=out, where=mass > 0)
    return out


def mass_diagnostics(mass: np.ndarray) -> dict:
    """What a pointwise 1/mass weighting will do: the share of points with no
    information, and how far the largest weight sits above the typical one."""
    mass = np.asarray(mass, dtype=np.float64).reshape(-1)
    live = mass[mass > 0]
    if live.size == 0:
        return {"zero_share": 1.0, "inv_max_over_median": float("nan")}
    inv = 1.0 / live
    return {"zero_share": float(np.mean(mass == 0)),
            "inv_max_over_median": float(inv.max() / np.median(inv))}


def _power(value, power: float):
    return value if power == 1.0 else value ** power


def factor_value(fs: FactorSpec, fields, coords, backend):
    """``fields[(var, orders)]`` and ``coords[axis]`` are arrays/tensors on
    the evaluation points; ``backend`` is ``numpy`` or ``torch``."""
    if fs.kind == "grid":
        return _power(coords[fs.axis], fs.power)
    if fs.kind == "trig":
        f = backend.sin if fs.func == "sin" else backend.cos
        return _power(f(fs.freq * coords[fs.axis]), fs.power)
    if fs.kind == "const":
        return fs.value ** fs.power
    return _power(fields[(fs.var, fs.orders)], fs.power)


def _product(factors, fields, coords, backend):
    value = 1.0
    for fs in factors:
        value = value * factor_value(fs, fields, coords, backend)
    return value


def residual(spec: EquationSpec, fields, coords, backend, with_mass: bool = False):
    """``sum_k c_k prod(f_k) + b - prod(target)``, and optionally the term
    mass ``|prod(target)| + sum_k |c_k prod(f_k)| + |b|``."""
    total = 0.0
    mass = 0.0
    for term in spec.terms:
        value = term.coeff * _product(term.factors, fields, coords, backend)
        total = total + value
        if with_mass:
            mass = mass + backend.abs(value) if hasattr(value, "shape") else mass + abs(value)
    total = total + spec.intercept
    target = _product(spec.target, fields, coords, backend)
    total = total - target
    if not with_mass:
        return total
    mass = mass + abs(spec.intercept) + (backend.abs(target) if hasattr(target, "shape")
                                         else abs(target))
    return total, mass
