"""PySINDy in Run a search: the PDEComp benchmark's own PySINDy, nothing else.

On PDEComp's datasets the user picks its ``fixed`` protocol (PDEComp's settings
for that dataset, the default) or ``native``. On any other data the native
protocol runs with PDEComp's default settings; it needs only the target time
derivative, taken from the known law, or chosen here for data without one.
"""
from __future__ import annotations

import streamlit as st

from epde_bench import pdecomp

PROTOCOL_TEXT = {
    "fixed": "fixed — shared NumPy derivatives and candidate library, with the optimizer PDEComp "
             "tuned for this dataset (the setting of PDEComp's published tables)",
    "native": "native — PySINDy's own finite differences and standard libraries",
}


def shape_of_arrays(arrays):
    """(kind, number of variables) of uploaded or generated arrays."""
    axes = [k for k in arrays if k.startswith("axis_") and k != "axis_names"]
    variables = [k for k in arrays if k.startswith("var_")]
    kind = {1: "ode" if len(variables) == 1 else "ode_system", 2: "pde_1d",
            3: "pde_2d", 4: "pde_3d"}.get(len(axes), "unknown")
    return kind, len(variables)


def unsupported(record_name=None, kind=None, variables=None):
    """Why PySINDy cannot run on this input, or None."""
    if record_name and pdecomp.supported(record_name):
        return None
    if record_name:
        from epde_bench.datasets import REGISTRY
        from support.discovery_lab.catalog import declared_equations
        kind = REGISTRY[record_name].kind
        variables = len(declared_equations(record_name)) or 1
    return pdecomp.unsupported_reason(kind, variables)


@st.cache_data(show_spinner=False)
def pdecomp_settings(record_name, protocol):
    """PDEComp's own settings for a record (``None`` = data outside its datasets)."""
    key = record_name if record_name and pdecomp.supported(record_name) else None
    return pdecomp.describe(key, protocol)


def settings_editor(record_name, protocol, key, kind=None, advice_context=None):
    """Editable PDEComp settings, prefilled with PDEComp's own; returns only the changes."""
    if kind is None and record_name:
        from epde_bench.datasets import REGISTRY
        kind = REGISTRY[record_name].kind
    base = pdecomp_settings(record_name, protocol)
    suggested = base
    if advice_context is not None:
        from support.discovery_lab.starting_settings import suggest_settings
        from support.discovery_lab.advice_state import save_report
        report = suggest_settings("pdecomp", base, advice_context, protocol=protocol,
                                  hosted=not advice_context.get("heavy", False))
        suggested = report["settings"]
        save_report(st.session_state, f"pdecomp:{record_name or 'data'}:{protocol}", report)
    opt = dict(base.get("optimizer") or {})
    prefix = f"lab_pdecomp_{protocol}_{key}_"
    if advice_context is not None:
        prefix += f"advice_{st.session_state.get('lab_advice_generation', 0)}_"
    if st.button("Reset to PDEComp's settings", key=f"pdecomp_reset_{protocol}_{key}"):
        saved = st.session_state.get("lab_widgets_saved", {})
        for k in [k for k in list(st.session_state) + list(saved) if k.startswith(prefix)]:
            st.session_state.pop(k, None)
            saved.pop(k, None)
        # A deliberate reset must not immediately reapply automatic suggestions.
        st.session_state[prefix + "use_defaults"] = True
        st.rerun()
    if st.session_state.get(prefix + "use_defaults"):
        suggested = base
    shown = {"type": opt.get("type", "STLSQ") if opt.get("type", "STLSQ") in pdecomp.OPTIMIZERS else "STLSQ",
             "threshold": float(opt.get("threshold", 0.1)), "max_iter": int(opt.get("max_iter", 10)),
             "alpha": float(opt.get("alpha", 0.05)),
             "normalize_columns": bool(opt.get("normalize_columns", False)),
             "coefficient_tol": float(opt.get("coefficient_tol", 0.0)),
             "coefficient_relative_tol": float(opt.get("coefficient_relative_tol", 0.0))}
    suggested_opt = suggested.get("optimizer") or {}
    initial = {name: type(value)(suggested_opt.get(name, value)) for name, value in shown.items()}
    a, b, c = st.columns(3)
    optimizer_kind = a.pills("Sparse optimizer", list(pdecomp.OPTIMIZERS), selection_mode="single",
                            default=initial["type"], key=prefix + "type") or initial["type"]
    edited = {"type": optimizer_kind}
    if optimizer_kind == "STLSQ":
        edited["threshold"] = float(b.number_input(
            "STLSQ threshold", 0.0, 1e4, initial["threshold"], format="%g",
            key=prefix + "threshold", help="Coefficients below it are removed at every iteration."))
    else:
        edited["max_iter"] = int(b.number_input(
            "FROLS terms (max_iter)", 1, 200, initial["max_iter"], key=prefix + "max_iter",
            help="Forward selection adds at most this many terms."))
    edited["alpha"] = float(c.number_input(
        "Ridge alpha", 0.0, 1e4, initial["alpha"], format="%g", key=prefix + "alpha"))
    a, b, c = st.columns(3)
    edited["normalize_columns"] = a.toggle("Normalize columns", initial["normalize_columns"],
                                           key=prefix + "normalize")
    edited["coefficient_tol"] = float(b.number_input(
        "Drop |coefficient| below", 0.0, 1e4, initial["coefficient_tol"], format="%g",
        key=prefix + "coef_tol"))
    edited["coefficient_relative_tol"] = float(c.number_input(
        "…or below this share of the largest", 0.0, 1.0, initial["coefficient_relative_tol"],
        format="%g", key=prefix + "coef_rel"))
    changes = {}
    optimizer = {k: v for k, v in edited.items() if shown[k] != v}
    if optimizer:
        changes["optimizer"] = optimizer
    a, b, c = st.columns(3)
    if protocol == "fixed":
        library = base.get("library") or {}
        common = base.get("common") or {}
        initial_library = suggested.get("library") or {}
        initial_common = suggested.get("common") or {}
        crop = int(a.number_input("Boundary crop, points", 0, 50, int(suggested.get("crop", 0)),
                                  key=prefix + "crop"))
        if crop != int(base.get("crop", 0)):
            changes["crop"] = crop
        if library.get("type") == "navier_stokes":
            st.caption("The Navier–Stokes library of PDEComp is specialised; only the optimizer "
                       "and the crop can be changed.")
        else:
            lib_changes = {}
            degree_default = int(library.get("data_fun_pow", common.get("data_fun_pow", 3)))
            degree_initial = int(initial_library.get("data_fun_pow", initial_common.get("data_fun_pow", 3)))
            degree = int(b.number_input("Polynomial degree of the library", 1, 4, degree_initial,
                                        key=prefix + "degree"))
            factors_default = int(library.get("equation_factors_max_number",
                                              common.get("equation_factors_max_number", 2)))
            factors_initial = int(initial_library.get("equation_factors_max_number",
                                                      initial_common.get("equation_factors_max_number", 2)))
            factors = c.pills("Feature interactions", [1, 2], selection_mode="single",
                              default=factors_initial, key=prefix + "factors",
                              format_func=lambda v: "Individual features" if v == 1
                              else "Features + pairwise products") or factors_default
            a, b, c = st.columns(3)
            bias_default = bool(library.get("include_bias", common.get("include_bias", True)))
            bias_initial = bool(initial_library.get("include_bias", initial_common.get("include_bias", True)))
            bias = a.toggle("Free coefficient (constant term)", bias_initial, key=prefix + "bias")
            orders_default = list(library.get("max_deriv_order", common.get("max_deriv_order", [2, 4])))
            orders_initial = list(initial_library.get("max_deriv_order", initial_common.get("max_deriv_order", [2, 4])))
            # The target (e.g. u_tt) comes from the same derivatives: the library must reach its order.
            target_time = max([int(t.get("order", 1)) for t in base.get("targets", [])
                               if t.get("axis", "t") == "t"] or [1])
            time_order = int(b.number_input("Highest time derivative", target_time, 2,
                                            max(target_time, int(orders_initial[0])),
                                            key=prefix + "time_order",
                                            help=f"At least {target_time}: the order of the target "
                                                 "time derivative."))
            orders = [time_order]
            if kind == "pde_1d" and len(orders_default) > 1:
                orders.append(int(c.number_input("Highest space derivative", 1, 4,
                                                  int(orders_initial[1]), key=prefix + "space_order")))
            vocabulary = list(pdecomp.CUSTOM_TOKENS.get(kind, ()))
            tokens_default = [t for t in library.get("custom_tokens", []) if t in vocabulary]
            tokens_initial = [t for t in initial_library.get("custom_tokens", []) if t in vocabulary]
            tokens = st.multiselect("Extra library terms", vocabulary, default=tokens_initial,
                                    key=prefix + "tokens",
                                    help="PDEComp's own extra terms; the defaults are PDEComp's choice "
                                         "for this dataset.")
            if degree != degree_default:
                lib_changes["data_fun_pow"] = degree
            if int(factors) != factors_default:
                lib_changes["equation_factors_max_number"] = int(factors)
            if bias != bias_default:
                lib_changes["include_bias"] = bias
            if orders != [int(o) for o in orders_default[:len(orders)]]:
                lib_changes["max_deriv_order"] = orders
            if tokens != tokens_default:
                lib_changes["custom_tokens"] = tokens
            if lib_changes:
                changes["library"] = lib_changes
    else:
        degree = int(a.number_input("Polynomial degree", 1, 4, int(suggested.get("polynomial_degree", 3)),
                                    key=prefix + "degree"))
        trim = int(b.number_input("Boundary trim, points", 0, 50, int(suggested.get("boundary_trim", 0)),
                                  key=prefix + "trim"))
        diff = base.get("differentiation") or {}
        initial_diff = suggested.get("differentiation") or {}
        order_default = int(diff.get("order", 4))
        order_initial = int(initial_diff.get("order", 4))
        order = c.pills("Finite-difference order", [2, 4], selection_mode="single",
                        default=order_initial if order_initial in (2, 4) else 4,
                        key=prefix + "diff_order") or order_default
        periodic_default = bool(diff.get("periodic", False))
        periodic = st.toggle("Periodic in space", bool(initial_diff.get("periodic", False)), key=prefix + "periodic",
                             help="Finite differences wrap around the space axis (periodic data).")
        if degree != int(base.get("polynomial_degree", 3)):
            changes["polynomial_degree"] = degree
        if trim != int(base.get("boundary_trim", 0)):
            changes["boundary_trim"] = trim
        diff_changes = {}
        if int(order) != order_default:
            diff_changes["order"] = int(order)
        if periodic != periodic_default:
            diff_changes["periodic"] = periodic
        if diff_changes:
            changes["differentiation"] = diff_changes
    changes = pdecomp.validate_overrides(changes, protocol, kind)
    st.caption("Changed from PDEComp's settings: " + ", ".join(
        f"{k}={v}" for k, v in changes.items()) if changes else "PDEComp's own settings, unchanged.")
    return changes


def render_settings(*, record_name, problem, arrays, source, advice_context=None):
    """Controls of the PySINDy run; returns protocols, target orders and settings changes."""
    st.markdown("**PySINDy · sparse regression as in the PDEComp benchmark**")
    overrides = {}
    if record_name and pdecomp.supported(record_name):
        st.caption("This record is one of PDEComp's datasets: its data, grid and window are PDEComp's, "
                   "and the settings below start from PDEComp's settings for it.")
        protocols = st.pills("Protocol", list(pdecomp.PROTOCOLS), selection_mode="multi",
                             default=["fixed"], key="lab_pdecomp_protocols",
                             format_func=lambda p: p) or []
        for p in protocols:
            with st.container(border=True):
                st.caption(PROTOCOL_TEXT[p])
                overrides[p] = settings_editor(record_name, p, record_name, advice_context=advice_context)
        st.caption(f"PDEComp dataset {pdecomp.PDECOMP_FILES[record_name]}. Source of the settings: "
                   f"[PDEComp at {pdecomp.PDECOMP_REV[:7]}](https://github.com/{pdecomp.PDECOMP_REPO}"
                   f"/blob/{pdecomp.PDECOMP_REV}/data/config.py)")
        return ({"protocols": list(protocols), "target_orders": None, "overrides": overrides}
                if protocols else None)
    st.caption("Native protocol of PDEComp, starting from its default settings: PySINDy's finite "
               "differences, polynomial, trigonometric and PDE libraries, sparse regression.")
    variables = list(problem.variables) if problem is not None else \
        [k[4:] for k in arrays if k.startswith("var_")]
    if problem is not None and problem.truth:
        orders = pdecomp.target_orders(problem)
        st.caption("Target: " + ", ".join(f"{v}_{'t' * k}" for v, k in orders.items())
                   + " — the time derivative on the left-hand side of the known law. "
                     "The law itself is not given to the method.")
        target_orders = None
    elif len(variables) == 1 and source != "Generate an ODE":
        order = st.pills("Target time derivative", [1, 2], default=1, selection_mode="single",
                         key="lab_pdecomp_target_order",
                         format_func=lambda k: "first (u_t)" if k == 1 else "second (u_tt)") or 1
        target_orders = {variables[0]: int(order)}
    else:
        st.caption("Target: the first time derivative of every variable.")
        target_orders = {v: 1 for v in variables}
    with st.container(border=True):
        kind = problem.kind if problem is not None else shape_of_arrays(arrays)[0]
        overrides["native"] = settings_editor(record_name, "native", record_name or "data", kind,
                                              advice_context=advice_context)
    return {"protocols": ["native"], "target_orders": target_orders, "overrides": overrides}


def review(record_name, settings):
    """The complete effective PySINDy settings for the review step."""
    key = record_name if record_name and pdecomp.supported(record_name) else None
    changes = settings.get("overrides") or {}
    return {
        "method": "PySINDy as run by the PDEComp benchmark",
        "source": f"{pdecomp.PDECOMP_REPO}@{pdecomp.PDECOMP_REV}",
        "starting point": ("PDEComp's settings for dataset " + pdecomp.PDECOMP_FILES[key] if key
                           else "PDEComp native defaults"),
        "target_orders": settings.get("target_orders") or "from the known law",
        "protocols": {p: {"changed": changes.get(p) or {},
                          "effective": pdecomp.describe(key, p, changes.get(p))}
                      for p in settings.get("protocols") or []},
    }

