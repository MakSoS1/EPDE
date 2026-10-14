"""Validated EPDE search controls, plus portable YAML configuration."""
from dataclasses import asdict

import numpy as np
import streamlit as st
import yaml

from epde_bench.config import deep_merge
from epde_bench.poly_options import polynomial_kwargs, validate_polynomial_options
from .common import search_config, variant_label, variants
from .config_io import parse_search_yaml, dump_search_yaml

DERIVATIVES = {"FD": "finite differences", "poly": "polynomial fits",
               "spectral": "spectral (experimental)"}


def _effort(pop, epochs):
    return {"Quick": (max(4, pop // 2), max(2, epochs // 2)),
            "Standard": (pop, epochs),
            "Thorough": (pop * 2, epochs * 2)}


def search_options(name):
    """Return (variant, seed, overrides, problem); block invalid configurations."""
    with st.expander("More options"):
        basic, expert = st.tabs(["Basic", "Expert"])
        with basic:
            a, b = st.columns([3, 1])
            variant = a.selectbox("method", variants(), format_func=variant_label,
                                  key=f"variant_{name}")
            seed = int(b.number_input("seed", 0, 999, 0, key=f"seed_{name}"))
        try:
            cfg, problem, resolved = search_config(name, variant)
        except (FileNotFoundError, ValueError, OSError) as exc:
            st.error(f"This record is unavailable: {exc}")
            return variant, seed, {}, None
        if cfg.get("method") != "epde":
            with basic:
                st.caption("This baseline uses its own fixed regression settings; EPDE options do not apply.")
            return variant, seed, {}, problem

        key = f"{name}_{variant}"
        prep, space, evo = resolved["preprocessing"], resolved["search_space"], resolved["evolution"]
        changes = {}
        with basic:
            now = prep.get("default_preprocessor_type", "FD")
            choices = list(DERIVATIVES) if now in DERIVATIVES else list(DERIVATIVES) + [now]
            method = st.radio("derivatives", choices, index=choices.index(now),
                              horizontal=True, format_func=lambda m: DERIVATIVES.get(m, m),
                              key=f"prep_{key}")
            if method != now:
                changes.setdefault("preprocessing", {})["default_preprocessor_type"] = method
            poly_params = None
            if method == "poly":
                st.caption("Chebyshev fitting: set an explicit polynomial degree and odd local window. "
                           "EPDE uses finite differences near the edges.")
                col1, col2 = st.columns(2)
                order_base = prep.get("max_deriv_order", [2])
                if isinstance(order_base, int):
                    order_base = [order_base]
                old_kw = prep.get("preprocessor_kwargs") or {}
                default_degree = old_kw.get("poly_order")
                default_degree = int(default_degree) if default_degree is not None else max(3, max(order_base) + 1)
                degree = int(col1.number_input("polynomial degree", min_value=1, max_value=12,
                                               value=min(12, default_degree), key=f"poly_degree_{key}"))
                win = int(old_kw.get("polynomial_window", 9))
                if win not in range(5, 34, 2):
                    win = 9
                window = int(col2.select_slider("polynomial window (points)",
                                                options=list(range(5, 34, 2)), value=win,
                                                key=f"poly_window_{key}"))
                poly_params = window, degree

            pop, epochs = int(evo.get("population_size", 16)), int(evo.get("training_epochs", 5))
            presets = _effort(pop, epochs)
            effort = st.radio("search effort", list(presets), index=1, horizontal=True,
                              key=f"effort_{key}",
                              format_func=lambda e: f"{e} ({presets[e][0]} candidates × {presets[e][1]} epochs)")
            a, b = st.columns(2)
            terms_now = int(space.get("equation_terms_max_number", 6))
            terms = a.slider("largest number of terms", 2, 15, min(15, terms_now),
                             key=f"terms_{key}")
            fmax = space.get("equation_factors_max_number")
            two = isinstance(fmax, dict) and max(fmax.get("factors_num", [1])) >= 2
            products = b.toggle("allow products such as u·uₓ", value=two, key=f"prod_{key}")
            if terms != terms_now:
                changes.setdefault("search_space", {})["equation_terms_max_number"] = terms
            if products != two:
                changes.setdefault("search_space", {})["equation_factors_max_number"] = (
                    {"factors_num": [1, 2], "probas": [0.65, 0.35]} if products else 1)
            if problem.dim >= 1:
                current_tokens = list(space.get("tokens", []) or [])
                has_inverse = any(t.get("family") == "inverse_space_coordinate" for t in current_tokens)
                inverse = st.toggle("include 1/x coordinate token (PDEDivide)",
                                    value=has_inverse, key=f"inverse_{key}",
                                    help="The spatial grid must have no x=0 point.")
                if inverse != has_inverse:
                    if inverse:
                        current_tokens.append({"family": "inverse_space_coordinate", "axis": 1})
                    else:
                        current_tokens = [t for t in current_tokens
                                          if t.get("family") != "inverse_space_coordinate"]
                    changes.setdefault("search_space", {})["tokens"] = current_tokens

        with expert:
            from epde.interface.search_config import METRIC_MENUS, SPARSITY_REGISTRY, load_search_config
            current = asdict(load_search_config(resolved).objectives)
            a, b = st.columns(2)
            picks = {}
            for column, field, label in [
                (a, "discrepancy_metric", "discrepancy"),
                (b, "second_objective", "second objective"),
                (a, "instability_metric", "instability estimator"),
            ]:
                menu = METRIC_MENUS[field]
                picks[field] = column.selectbox(label, menu, index=menu.index(current[field]),
                                                key=f"{field}_{key}")
            sparsity_now = resolved.get("objectives", {}).get("sparsity_cls", "vwsr")
            sparsity_now = sparsity_now if sparsity_now in SPARSITY_REGISTRY else "vwsr"
            picks["sparsity_cls"] = b.selectbox("sparsity", SPARSITY_REGISTRY,
                                               index=SPARSITY_REGISTRY.index(sparsity_now),
                                               key=f"sparsity_{key}")
            current["sparsity_cls"] = sparsity_now
            for field, value in picks.items():
                if value != current[field]:
                    changes.setdefault("objectives", {})[field] = value
            dpow_now = int(space.get("data_fun_pow", 1))
            dpow = a.slider("highest power of a variable", 1, 4, min(4, dpow_now),
                            key=f"dpow_{key}")
            if dpow != dpow_now:
                changes.setdefault("search_space", {})["data_fun_pow"] = dpow
            orders = prep["max_deriv_order"]
            orders = list(orders) if isinstance(orders, (list, tuple)) else [int(orders)]
            new_orders = [
                int(col.number_input(f"highest derivative in {axis}", 1, 6, min(6, int(k)),
                                     key=f"ord_{axis}_{key}"))
                for col, axis, k in zip(st.columns(len(orders)), problem.axis_names, orders)
            ]
            if new_orders != orders:
                changes.setdefault("preprocessing", {})["max_deriv_order"] = new_orders
            if poly_params is not None:
                try:
                    kw = polynomial_kwargs(poly_params[0], poly_params[1],
                                            new_orders, problem.shape)
                except ValueError as exc:
                    st.error(str(exc))
                    st.stop()
                if kw != (prep.get("preprocessor_kwargs") or {}):
                    changes.setdefault("preprocessing", {})["preprocessor_kwargs"] = kw
            exact = st.toggle("set candidates and epochs exactly", key=f"exact_{key}")
            if exact:
                a, b = st.columns(2)
                pop_e = a.slider("candidates", 4, 96, min(96, max(4, presets[effort][0])),
                                 step=4, key=f"pop_{key}")
                epochs_e = b.slider("epochs", 1, 50, min(50, presets[effort][1]),
                                    key=f"epochs_{key}")
            else:
                pop_e, epochs_e = presets[effort]
            if pop_e != pop:
                changes.setdefault("evolution", {})["population_size"] = pop_e
            if epochs_e != epochs:
                changes.setdefault("evolution", {})["training_epochs"] = epochs_e

            st.markdown("**Import/export reproducible settings**")
            uploaded = st.file_uploader("Import search overrides (YAML)",
                                        type=["yaml", "yml"], key=f"import_{key}")
            imported = {}
            if uploaded is not None:
                try:
                    imported = parse_search_yaml(uploaded.getvalue())["search"]
                    st.info("Imported settings are applied first; the visible controls above win.")
                except ValueError as exc:
                    st.error(str(exc))
                    st.stop()
            combined = deep_merge(imported, changes)
            overrides = {"search": combined}
            effective = deep_merge(resolved, combined)
            try:
                load_search_config(effective)
                validate_polynomial_options(effective, problem.shape)
                inverses = [t for t in effective.get("search_space", {}).get("tokens", [])
                            if t.get("family") == "inverse_space_coordinate"]
                if inverses:
                    if problem.dim < 1 or np.any(np.isclose(np.asarray(problem.grids[1]), 0.0, atol=1e-12)):
                        raise ValueError("1/x requires a spatial grid with no zero coordinates.")
            except (ValueError, TypeError, KeyError, IndexError) as exc:
                st.error(f"Invalid final settings: {exc}")
                st.stop()
            st.caption("Reproducibility: effective overrides (applied to stored defaults)")
            st.code(yaml.safe_dump(overrides, sort_keys=False) if combined else "# none",
                    language="yaml")
            st.download_button("Export settings YAML", dump_search_yaml(overrides),
                               file_name=f"{name}_{variant}_search.yaml",
                               mime="application/x-yaml", key=f"export_{key}")
    return variant, seed, (overrides if overrides["search"] else {}), problem
