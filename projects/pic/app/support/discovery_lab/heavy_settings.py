"""Metadata-only scientific controls for large original ODE/PDE records.

Never loads a grid/NetCDF/MAT file on the public Streamlit machine.
"""
from __future__ import annotations

import copy

import streamlit as st

from epde_bench.config import load_config
from support.discovery_lab.sindy_settings import validate_sindy_options


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
    with st.expander("Advanced EPDE: effective scientific configuration"):
        import yaml
        st.caption("The full dataset-specific configuration is preserved. "
                   "The following declarative YAML is included in your download.")
        st.code(yaml.safe_dump({"search": search}, sort_keys=False), language="yaml")
    return {"method": "epde", "variant": "default", "overrides": {"search": search}}


def render_heavy_sindy(name):
    from epde_bench.baselines import unsupported_reason
    reason = unsupported_reason(name)
    if reason:
        st.error("PySINDy's current scientific adapter cannot represent this equation: " + reason + ". Deselect PySINDy in step 1 to export EPDE alone.")
        return None
    cfg = load_config(name, "pysindy")
    search = copy.deepcopy(cfg["search"])
    options = copy.deepcopy(cfg.get("pysindy", {}))
    st.caption("PySINDy sparse regression; no EPDE population/epoch settings are applied.")
    mode = st.pills("Differentiation", ["epde", "finite_difference", "smoothed_finite_difference"],
                    default=options.get("differentiation", "epde"),
                    selection_mode="single", key="portable_sindy_diff_" + name)
    normalization = st.pills("Feature normalization", ["auto", "on", "off"],
                             default=options.get("normalize_columns", "auto"),
                             selection_mode="single", key="portable_sindy_normalize_" + name)
    threshold_mode = st.pills("STLSQ threshold", ["auto", "manual"], default="auto",
                              selection_mode="single", key="portable_sindy_mode_" + name)
    if threshold_mode == "manual":
        threshold = st.number_input("Fixed STLSQ threshold", min_value=1e-6,
                                     max_value=100.0, value=0.01, format="%.6f",
                                     key="portable_sindy_threshold_" + name)
    else:
        threshold = "auto"
    a, b = st.columns(2)
    factors = int(a.number_input("Maximum feature factors", min_value=1, max_value=2,
                                 value=int(options.get("max_terms_product", 2)),
                                 key="portable_sindy_factors_" + name))
    holdout = b.pills("Held-out time fraction", [0.10, 0.20, 0.30],
                      default=0.20, selection_mode="single",
                      key="portable_sindy_holdout_" + name,
                      format_func=lambda n: f"{n:.0%}")
    options.update(differentiation=mode or "epde",
                   normalize_columns=normalization or "auto",
                   threshold=threshold, max_terms_product=factors,
                   validation_fraction=holdout or 0.20)
    options = validate_sindy_options(options)
    with st.expander("Advanced PySINDy: effective scientific configuration"):
        import yaml
        st.code(yaml.safe_dump({"search": search, "pysindy": options},
                               sort_keys=False), language="yaml")
    return {"method": "pysindy", "variant": "pysindy",
            "overrides": {"search": search, "pysindy": options}}
