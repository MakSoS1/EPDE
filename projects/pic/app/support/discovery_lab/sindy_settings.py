"""Method-specific PySINDy scientific settings for unified Run a search.

The stock PDEComp/native PySINDy baseline uses an EPDE-compatible candidate
library but its STLSQ optimizer, threshold sweep and held-out selection are
distinct. Do not display EPDE population, epochs, Pareto or sparsity parameters
as PySINDy configuration; none of those affect PySINDy fitting.
"""
from __future__ import annotations

import copy
import math

import streamlit as st

from epde_bench.config import load_config
from support.discovery_lab.lab_settings import _custom_base, validate_public_search


ALLOWED_DIFF = ("epde", "finite_difference", "smoothed_finite_difference")
ALLOWED_NORMALIZE = ("auto", "on", "off")


def validate_sindy_options(options, *, supplied_derivatives=False):
    """Reject unsupported/unbounded method-specific scientific options."""
    if not isinstance(options, dict):
        raise ValueError("PySINDy options must be a mapping")
    allowed = {"differentiation", "threshold", "max_terms_product",
               "normalize_columns", "validation_fraction"}
    if set(options) - allowed:
        raise ValueError("Unknown PySINDy option: " + ", ".join(sorted(set(options)-allowed)))
    differentiation = options.get("differentiation", "epde")
    if differentiation not in ALLOWED_DIFF:
        raise ValueError("Unknown PySINDy derivative estimator")
    if supplied_derivatives and differentiation != "epde":
        raise ValueError("The dataset supplies exact derivatives; use its supplied derivatives")
    threshold = options.get("threshold", "auto")
    if threshold != "auto":
        if type(threshold) not in (float, int) or not math.isfinite(threshold) or not 1e-6 <= threshold <= 100:
            raise ValueError("PySINDy STLSQ threshold must be auto or between 1e-6 and 100")
    max_factors = options.get("max_terms_product", 2)
    if type(max_factors) is not int or max_factors not in (1, 2):
        raise ValueError("PySINDy library products must have 1 or 2 factors")
    if options.get("normalize_columns", "auto") not in ALLOWED_NORMALIZE:
        raise ValueError("Unknown PySINDy normalization setting")
    fraction = options.get("validation_fraction", 0.20)
    if type(fraction) not in (int, float) or not math.isfinite(fraction) or not 0.10 <= fraction <= 0.40:
        raise ValueError("PySINDy validation fraction must be 10–40%")
    return {
        "differentiation": differentiation,
        "threshold": threshold,
        "max_terms_product": max_factors,
        "normalize_columns": options.get("normalize_columns", "auto"),
        "validation_fraction": float(fraction),
    }


def _initial_configuration(record_name, arrays, space, source):
    if record_name is not None:
        cfg = load_config(record_name, "pysindy")
        return copy.deepcopy(cfg["search"]), copy.deepcopy(cfg.get("pysindy", {}))
    axes = sorted(int(k[5:]) for k in arrays if k.startswith("axis_") and k != "axis_names")
    base = _custom_base(len(axes) - 1, generated=source == "Generate an ODE")
    tokens = list(base["search_space"].get("tokens") or [])
    if not space.get("time_terms", True):
        tokens = [token for token in tokens if token.get("family") != "grid"]
    if space.get("frequency") is not None:
        tokens.append({"family": "fixed_trigonometric",
                       "freq": float(space["frequency"]), "meaningful": True})
    base["search_space"]["tokens"] = tokens
    return base, {"differentiation": "epde", "threshold": "auto",
                  "max_terms_product": 2}


def render_sindy_settings(*, record_name, problem, arrays, space, source):
    """One independent PySINDy configuration, directly consumed by native adapter."""
    if record_name is None and arrays is None:
        return None
    search, initial = _initial_configuration(record_name, arrays, space, source)
    shape = tuple(problem.shape) if record_name is not None else tuple(
        len(arrays[f"axis_{i}"]) for i in range(
            sum(k.startswith("axis_") and k != "axis_names" for k in arrays)))
    supplied = bool(problem is not None and problem.derivs is not None)
    key = f"sindy_{source}_{record_name or 'custom'}"

    st.caption("Sparse regression uses its own differentiation and STLSQ settings; "
               "EPDE population and epochs do not apply.")
    mode_labels = {
        "epde": "Dataset / EPDE derivatives",
        "finite_difference": "Finite differences",
        "smoothed_finite_difference": "Smoothed finite differences",
    }
    methods = ("epde",) if supplied else ALLOWED_DIFF
    a, b = st.columns([3, 2])
    method = a.pills(
        "Derivative estimator", methods,
        default=initial.get("differentiation", "epde")
                if initial.get("differentiation", "epde") in methods else "epde",
        selection_mode="single", format_func=mode_labels.get,
        key=key + "_derivatives",
        help="Dataset/EPDE derivatives use the original numerical preprocessing. "
             "Independent methods use PySINDy's finite differences.")
    if supplied:
        b.info("Exact derivatives supplied by the scientific dataset; "
               "independent estimation is not valid for this record.")
    else:
        b.caption("Preprocessing is PySINDy-specific and is saved with each run.")
    options = {"differentiation": method or "epde"}

    pool = search["search_space"]
    with st.expander("Advanced PySINDy settings · candidate library", expanded=False):
        x, y, z = st.columns(3)
        pool["data_fun_pow"] = int(x.slider(
            "Maximum variable power", 1, 4,
            value=min(4, int(pool.get("data_fun_pow", 3))), key=key + "_field"))
        pool["deriv_fun_pow"] = int(y.slider(
            "Maximum derivative power", 1, 4,
            value=min(4, int(pool.get("deriv_fun_pow", 2))), key=key + "_derivpower"))
        pool["equation_terms_max_number"] = int(z.slider(
            "Maximum non-zero terms", 2, 15,
            value=min(15, int(pool.get("equation_terms_max_number", 10))),
            key=key + "_terms"))
        factor = st.pills(
            "Feature interactions", [1, 2], default=int(initial.get("max_terms_product", 2)),
            selection_mode="single", format_func=lambda value: (
                "Individual features" if value == 1 else "Features + pairwise products"),
            key=key + "_interactions")
        options["max_terms_product"] = factor or 2
        pool["equation_factors_max_number"] = (
            1 if options["max_terms_product"] == 1
            else {"factors_num": [1, 2], "probas": [0.65, 0.35]})

    st.markdown("**Basic PySINDy settings**")
    a, b = st.columns(2)
    mode = a.pills("STLSQ sparsity threshold",
                   ["auto", "manual"], default="auto",
                   selection_mode="single",
                   format_func=lambda x: "Automatic search" if x == "auto" else "Fixed value",
                   key=key + "_threshold_mode")
    if mode == "manual":
        threshold = a.number_input(
            "Threshold (STLSQ)", min_value=1e-6, max_value=100.0,
            value=0.01, format="%.6f", key=key + "_threshold")
    else:
        threshold = "auto"
        a.caption("Original protocol: sweep 49 thresholds, keep the sparsest "
                  "model within validation-error tolerance.")
    normalization = b.pills(
        "Normalize feature columns",
        list(ALLOWED_NORMALIZE), default=initial.get("normalize_columns", "auto"),
        selection_mode="single", key=key + "_normalization",
        format_func=lambda x: {"auto": "Try both", "on": "On", "off": "Off"}[x])
    options.update(threshold=threshold, normalize_columns=normalization or "auto")

    holdout = st.pills(
        "Validation: final fraction of the time series",
        [0.10, 0.20, 0.30], default=0.20, selection_mode="single",
        format_func=lambda x: f"{x:.0%} held out", key=key + "_holdout")
    options["validation_fraction"] = holdout if holdout is not None else 0.20
    options = validate_sindy_options(options, supplied_derivatives=supplied)
    validate_public_search(search, shape, override=False)

    return {"search": search, "pysindy": options}
