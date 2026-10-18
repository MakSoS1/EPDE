"""Metadata-only scientific controls for large original ODE/PDE records.

Never loads a grid/NetCDF/MAT file on the public Streamlit machine.
"""
from __future__ import annotations

import copy

import streamlit as st

from epde_bench.config import load_config


def render_heavy_epde(name):
    cfg = load_config(name, "default")
    search = copy.deepcopy(cfg["search"])
    evo = search.setdefault("evolution", {})
    sp = search.setdefault("search_space", {})
    st.caption("Original EPDE pipeline; its native worker will load the full dataset.")
    a, b, c = st.columns(3)
    evo["population_size"] = int(a.number_input(
        "Population / candidates", 4, 512,
        value=min(512, max(4, int(evo.get("population_size", 16)))), step=4,
        key="portable_epde_population_" + name))
    evo["training_epochs"] = int(b.number_input(
        "Evolution epochs", 1, 200,
        value=min(200, max(1, int(evo.get("training_epochs", 5)))),
        key="portable_epde_epochs_" + name))
    sp["equation_terms_max_number"] = int(c.number_input(
        "Maximum terms", 2, 15,
        value=min(15, max(2, int(sp.get("equation_terms_max_number", 10)))),
        key="portable_epde_terms_" + name))
    st.caption("The complete dataset-specific EPDE YAML is shown on the review screen.")
    return {"method": "epde", "variant": "default", "overrides": {"search": search}}


def render_heavy_pysindy(name):
    """PySINDy as the PDEComp benchmark runs it, for a record computed locally."""
    from epde_bench import pdecomp
    from epde_bench.datasets import REGISTRY
    spec = REGISTRY[name]
    reason = None if pdecomp.supported(name) else pdecomp.unsupported_reason(spec.kind, len(_variables(name)))
    if reason:
        st.error("PySINDy cannot represent this equation: " + reason +
                 ". Deselect PySINDy in step 1 to export EPDE alone.")
        return None
    from support.discovery_lab.pysindy_method import settings_editor
    protocols = pdecomp.protocols_for(name)
    protocol = st.pills("PDEComp protocol", list(protocols), default=protocols[0],
                        selection_mode="single", key="portable_pdecomp_protocol_" + name) or protocols[0]
    st.caption("PDEComp's own PySINDy code; the settings start from "
               + ("PDEComp's settings for this dataset." if pdecomp.supported(name)
                  else "PDEComp's native defaults."))
    changes = settings_editor(name, protocol, "heavy_" + name)
    return {"method": "pdecomp", "variant": f"pysindy_pdecomp_{protocol}",
            "overrides": {"pdecomp": {"overrides": changes}} if changes else {}}


def _variables(name):
    """Variable names declared by the record's loader, without loading its data."""
    from support.discovery_lab.catalog import declared_equations
    return declared_equations(name) or ["u"]
