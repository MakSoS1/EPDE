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


def render_settings(*, record_name, problem, arrays, source):
    """Controls of the PySINDy run; returns ``{"protocols": [...], "target_orders": ...}``."""
    st.markdown("**PySINDy · sparse regression as in the PDEComp benchmark**")
    if record_name and pdecomp.supported(record_name):
        st.caption("This record is one of PDEComp's datasets: its data, grid and window are PDEComp's, "
                   "and PySINDy runs with PDEComp's settings for it.")
        protocols = st.pills("Protocol", list(pdecomp.PROTOCOLS), selection_mode="multi",
                             default=["fixed"], key="lab_pdecomp_protocols",
                             format_func=lambda p: p) or []
        for p in protocols:
            st.caption(PROTOCOL_TEXT[p])
        st.caption(f"PDEComp dataset {pdecomp.PDECOMP_FILES[record_name]}. Settings: "
                   f"[PDEComp at {pdecomp.PDECOMP_REV[:7]}](https://github.com/{pdecomp.PDECOMP_REPO}"
                   f"/blob/{pdecomp.PDECOMP_REV}/data/config.py)")
        return {"protocols": list(protocols), "target_orders": None} if protocols else None
    st.caption("Native protocol of PDEComp with its default settings: PySINDy's finite differences, "
               "polynomial, trigonometric and PDE libraries, STLSQ. PDEComp has no settings tuned for "
               "these data, so none are invented here.")
    variables = list(problem.variables) if problem is not None else \
        [k[4:] for k in arrays if k.startswith("var_")]
    if problem is not None and problem.truth:
        orders = pdecomp.target_orders(problem)
        st.caption("Target: " + ", ".join(f"{v}_{'t' * k}" for v, k in orders.items())
                   + " — the time derivative on the left-hand side of the known law. "
                     "The law itself is not given to the method.")
        return {"protocols": ["native"], "target_orders": None}
    if len(variables) == 1 and source != "Generate an ODE":
        order = st.pills("Target time derivative", [1, 2], default=1, selection_mode="single",
                         key="lab_pdecomp_target_order",
                         format_func=lambda k: "first (u_t)" if k == 1 else "second (u_tt)") or 1
        return {"protocols": ["native"], "target_orders": {variables[0]: int(order)}}
    st.caption("Target: the first time derivative of every variable.")
    return {"protocols": ["native"], "target_orders": {v: 1 for v in variables}}


def review(record_name, settings):
    """A plain summary of the PySINDy run for the review step."""
    return {
        "method": "PySINDy as run by the PDEComp benchmark",
        "protocols": settings.get("protocols"),
        "settings": ("PDEComp's settings for dataset " + pdecomp.PDECOMP_FILES[record_name]
                     if record_name and pdecomp.supported(record_name)
                     else "PDEComp native defaults"),
        "target_orders": settings.get("target_orders") or "from the known law",
        "source": f"{pdecomp.PDECOMP_REPO}@{pdecomp.PDECOMP_REV}",
    }

