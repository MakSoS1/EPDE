"""Reusable Run-a-search settings inside the single Discovery Lab.

Built-in presets come from the ORIGINAL layered YAML, not an invented global
parameter grid. All public modifications are declarative and bounded.
"""
from __future__ import annotations

import copy
import os

import streamlit as st
import yaml

from epde_bench.config import deep_merge, load_config, read_yaml, resolve_for_problem, variants
from epde_bench.paths import CONFIG_DIR
from epde_bench.poly_options import polynomial_kwargs, validate_polynomial_options
from support.config_io import dump_search_yaml, parse_search_yaml

DERIVATIVES = {"FD": "Finite differences", "poly": "Polynomial fitting",
               "spectral": "Spectral (requires suitable boundary assumptions)"}
PUBLIC_TOKEN_FAMILIES = {
    "grid", "constant", "trigonometric", "fixed_trigonometric",
    "inverse_space_coordinate", "data_polynomials", "phased_sine_1d",
}
EPDE_VARIANTS = tuple(
    variant for variant, cfg in variants().items()
    if cfg.get("method", "epde") == "epde"
)
MAX_TOKENS = 12


def validate_public_search(search, shape, *, override=False):
    """Reject unbounded/unsupported browser-supplied configurations.

    Exact EPDE's own structural validation runs after this public resource gate.
    No arbitrary import paths, callables, GPU solvers or operator mappings.
    """
    if not isinstance(search, dict):
        raise ValueError("Search settings must be a mapping")
    from epde.interface.search_config import load_search_config
    p = search.get("preprocessing") or {}
    sp = search.get("search_space") or {}
    ev = search.get("evolution") or {}
    dm = search.get("domain") or {}
    runtime = search.get("runtime") or {}
    solver = search.get("solver") or {}
    objectives = search.get("objectives") or {}

    pop = int(ev.get("population_size", 16))
    epochs = int(ev.get("training_epochs", 5))
    pop_limit, epoch_limit = (512, 200) if override else (96, 50)
    if not 2 <= pop <= pop_limit or not 1 <= epochs <= epoch_limit:
        raise ValueError(f"EPDE budget must be population 2–{pop_limit}, "
                         f"epochs 1–{epoch_limit}")
    if not 1 <= int(sp.get("data_fun_pow", 3)) <= 4:
        raise ValueError("Variable degree must be 1–4")
    if not 1 <= int(sp.get("deriv_fun_pow", 2)) <= 4:
        raise ValueError("Derivative degree must be 1–4")
    if not 2 <= int(sp.get("equation_terms_max_number", 10)) <= 15:
        raise ValueError("Maximum terms must be 2–15")
    factors = sp.get("equation_factors_max_number", 2)
    if isinstance(factors, dict):
        fnums = factors.get("factors_num", [1, 2])
        if not isinstance(fnums, (tuple, list)) or not fnums or any(
                type(v) is not int or v < 1 or v > 2 for v in fnums):
            raise ValueError("Only one or two factors per term are permitted")
    elif type(factors) is not int or factors < 1 or factors > 2:
        raise ValueError("Only one or two factors per term are permitted")
    tokens = sp.get("tokens") or []
    if not isinstance(tokens, list) or len(tokens) > MAX_TOKENS:
        raise ValueError("Token pool exceeds the public limit")
    if any(not isinstance(x, dict) or x.get("family") not in PUBLIC_TOKEN_FAMILIES
           for x in tokens):
        raise ValueError("Only approved declarative token families may be used")
    # A 1/x token is invalid on a spatial axis containing zero: enforce the
    # full numerical check in the data-bound worker as well as in the UI.
    if any(x.get("family") == "inverse_space_coordinate" for x in tokens) and len(shape) < 2:
        raise ValueError("1/x requires a spatial coordinate axis")
    if p.get("default_preprocessor_type", "FD") not in DERIVATIVES:
        raise ValueError("Unsupported public differentiator")
    orders = p.get("max_deriv_order", "auto")
    if orders != "auto":
        if isinstance(orders, int):
            orders = [orders]
        if not isinstance(orders, (tuple, list)) or len(orders) != len(shape) or any(
                not isinstance(n, int) or not 1 <= n <= 6 for n in orders):
            raise ValueError("Derivative orders must be 1–6, one per data axis")
        if any(int(n) <= k for n,k in zip(shape,orders)):
            raise ValueError("Derivative orders exceed grid size")
    boundary = dm.get("boundary_width", "auto")
    if boundary != "auto":
        widths = boundary if isinstance(boundary, (tuple, list)) else [boundary] * len(shape)
        if len(widths) != len(shape) or any(type(w) is not int or
                w < 0 or 2 * w + 8 >= n for w, n in zip(widths, shape)):
            raise ValueError("Boundary trimming leaves too few interior values")
    if bool(solver.get("use_solver", False)):
        raise ValueError("Solver-based EPDE evaluation is disabled on the hosted demo")
    if "operators" in ev and ev["operators"]:
        raise ValueError("Custom operator mappings are not supported in the public demo")
    cache_limit = 16 if override else 4
    if not 1 <= float(runtime.get("memory_for_cache", 2)) <= cache_limit:
        raise ValueError(f"Tensor cache limit: 1–{cache_limit}% memory")
    if objectives.get("research_objective_metric") or objectives.get("research_regularizer_metric"):
        raise ValueError("Research-only objective channels are not public options")
    validate_polynomial_options(search, shape)
    # Original parser catches unknown fields and invalid metric/optimizer identifiers.
    load_search_config(search)
    return search


def _custom_base(dim, *, generated=False):
    cfg = copy.deepcopy(read_yaml(CONFIG_DIR / "_base.yaml")["search"])
    cfg["domain"] = {"boundary_width": "auto"}
    cfg["preprocessing"] = {
        "default_preprocessor_type": "FD",
        "preprocessor_kwargs": {},
        "max_deriv_order": [1] + [2] * dim if generated else [2] + [3] * dim,
    }
    cfg["search_space"] = deep_merge(cfg["search_space"], {
        "data_fun_pow": 3, "deriv_fun_pow": 1,
        "equation_terms_max_number": 8,
        "equation_factors_max_number": {
            "factors_num": [1, 2], "probas": [0.65, 0.35],
        },
        "tokens": [{"family": "grid", "labels": ["x"], "max_power": 2}],
    })
    cfg["evolution"] = deep_merge(cfg.get("evolution", {}), {
        "population_size": 12, "training_epochs": 4,
    })
    cfg["solver"] = deep_merge(cfg.get("solver", {}), {
        "use_solver": False, "device": "cpu",
    })
    return cfg


def _profile_summary(search, description):
    pre, pool, evo = (search.get(k) or {} for k in (
        "preprocessing", "search_space", "evolution"))
    st.caption(
        f"{description} · {evo.get('population_size')} candidates × "
        f"{evo.get('training_epochs')} epochs · "
        f"{pre.get('default_preprocessor_type', 'FD')} derivatives · "
        f"up to {pool.get('equation_terms_max_number')} terms"
    )


def render_settings(*, record_name, problem, arrays, space, source, methods,
                    noise, seed, label, origin, reference):
    """Draw one common settings editor. Return (variant, overrides, custom_settings, label).

    A stored record's *default* loads config layers without synthetic budget
    overrides. An uploaded record starts from the universal conservative profile.
    """
    built_in = record_name is not None
    if built_in:
        c1, c2 = st.columns([2, 3])
        variant = c1.pills("EPDE profile", EPDE_VARIANTS, default="default",
                           selection_mode="single",
                           key=f"lab_variant_{record_name}",
                           help="Choose a reproducible EPDE variant. Dataset-specific "
                                "YAML is loaded automatically.") or "default"
        cfg = load_config(record_name, variant)
        base = copy.deepcopy(cfg["search"])
        shape = tuple(problem.shape)
        if variant != "default":
            c2.info(f"Selected EPDE variant: {variant}. Other method outputs remain "
                    "method-specific; their score axes are not directly comparable.")
        else:
            c2.caption("The original repository's per-equation profile is loaded "
                       "automatically from the layered YAML settings. "
                       "It is a reproducible baseline, not a proven optimal model.")
        if "pysindy" in methods:
            st.caption("PySINDy uses the same data, derivative estimator and candidate "
                       "features in this experiment; its STLSQ optimizer and "
                       "selection procedure remain separate.")
        context = f"Saved record: configs/{record_name}.yaml + {variant} variant"
    else:
        variant = "default"
        if arrays is None:
            return variant, {}, None, "No numerical input"
        shape = tuple(int(arrays[f"axis_{k}"].size)
                      for k in range(sum(k.startswith("axis_") and k != "axis_names"
                                       for k in arrays)))
        dim = len(shape) - 1
        base = _custom_base(dim, generated=source == "Generate an ODE")
        tokens = list(base["search_space"].get("tokens") or [])
        if not space.get("time_terms", True):
            tokens = [x for x in tokens if x.get("family") != "grid"]
        if space.get("frequency") is not None:
            tokens.append({"family": "fixed_trigonometric",
                           "freq": float(space["frequency"]), "meaningful": True})
        base["search_space"]["tokens"] = tokens
        context = "Universal bounded starting profile for new data"
        st.caption("General-purpose starting settings, not tuned to your unknown equation. "
                   "Start with Preset, then switch to Advanced for further controls.")

    _profile_summary(base, context)
    search = copy.deepcopy(base)
    prep, sp, evo = (search.setdefault(k, {}) for k in (
        "preprocessing", "search_space", "evolution"))
    objectives = search.setdefault("objectives", {})
    domain = search.setdefault("domain", {})
    resource = search.setdefault("runtime", {})

    st.markdown("**Search budget and candidate library**")
    preset = st.segmented_control(
        "Search budget",
        ["Preset", "Quick", "Thorough", "Custom"],
        default="Preset",
        key=f"lab_budget_{source}_{record_name}_{variant}",
        help="Preset is loaded from this equation's stored YAML. "
             "Quick and Thorough change the number of candidates and epochs.",
    ) or "Preset"
    original_pop, original_epochs = int(evo.get("population_size", 16)), int(
        evo.get("training_epochs", 5))
    derived_budget = {
        "Preset": (original_pop, original_epochs),
        "Quick": (max(4, original_pop // 2),
                                  max(2, original_epochs // 2)),
        "Thorough": (min(96, original_pop * 2),
                                     min(50, original_epochs * 2)),
        "Custom": (original_pop, original_epochs),
    }
    local_full = os.environ.get("EPDE_LOCAL_FULL_RUN") == "1"
    pop_limit, epoch_limit = (512, 200) if local_full else (96, 50)
    dpop, dep = derived_budget[preset]
    if not 4 <= int(dpop) <= pop_limit or not 1 <= int(dep) <= epoch_limit:
        st.warning(f"Budget exceeds the selected environment limits "
                   f"({pop_limit} candidates, {epoch_limit} epochs); select "
                   "the original research runner for larger experiments.")
        dpop, dep = max(4, min(pop_limit, int(dpop))), max(1, min(epoch_limit, int(dep)))
    a, b, c = st.columns(3)
    pop = a.number_input("Population / candidates", min_value=4, max_value=pop_limit,
                         value=dpop, step=4, key=f"lab_pop_{source}_{record_name}_{variant}_{preset}")
    epochs = b.number_input("Evolution epochs", min_value=1, max_value=epoch_limit,
                            value=dep, step=1,
                            key=f"lab_epochs_{source}_{record_name}_{variant}_{preset}")
    terms = c.slider("Maximum terms in an equation", min_value=2, max_value=15,
                     value=min(15, int(sp.get("equation_terms_max_number", 10))),
                     key=f"lab_terms_{source}_{record_name}_{variant}")
    evo.update(population_size=int(pop), training_epochs=int(epochs))
    sp["equation_terms_max_number"] = int(terms)
    st.caption("Population and epochs control EPDE search cost. The term bound also "
               "affects the shared SINDy candidate selection protocol.")

    detail = st.segmented_control("Configuration detail", ["Essential", "Advanced"],
                                  default="Essential",
                                  key=f"lab_detail_{source}_{record_name}_{variant}")
    with st.container(border=True):
        current = prep.get("default_preprocessor_type", "FD")
        now = st.segmented_control("Numerical derivatives", list(DERIVATIVES),
                                   default=current if current in DERIVATIVES else "FD",
                                   format_func=DERIVATIVES.get,
                                   key=f"lab_diff_{source}_{record_name}_{variant}") or "FD"
        if now != current:
            prep["preprocessor_kwargs"] = {}
        prep["default_preprocessor_type"] = now
        orders = prep.get("max_deriv_order", "auto")
        if orders == "auto":
            orders = [2] + [4] * (len(shape) - 1)
        if isinstance(orders, int):
            orders = [orders]
        selected_orders = []
        axes = list(getattr(problem, "axis_names", None)
                    or arrays.get("axis_names", []) if arrays is not None
                    else getattr(problem, "axis_names", []))
        for i, order in enumerate(orders):
            label_axis = str(axes[i]) if i < len(axes) else ("t" if i == 0 else f"x{i}")
            selected_orders.append(int(st.number_input(
                f"Maximum derivative order on {label_axis}", 1, 6,
                min(6, int(order)), key=f"lab_order_{source}_{record_name}_{variant}_{i}")))
        prep["max_deriv_order"] = selected_orders
        if now == "poly":
            kw = prep.get("preprocessor_kwargs") or {}
            a, b = st.columns(2)
            default_degree = int(kw.get("poly_order") or max(3, max(selected_orders) + 1))
            degree = a.number_input("Polynomial degree", min_value=1, max_value=12,
                                    value=min(12, default_degree),
                                    key=f"lab_poly_deg_{source}_{record_name}_{variant}")
            old_win = int(kw.get("polynomial_window", 11))
            opts = list(range(5, 34, 2))
            window = b.select_slider("Polynomial window (odd samples)", options=opts,
                                     value=old_win if old_win in opts else 11,
                                     key=f"lab_poly_win_{source}_{record_name}_{variant}")
            prep["preprocessor_kwargs"] = polynomial_kwargs(
                int(window), int(degree), selected_orders, shape)
        elif now != current:
            prep["preprocessor_kwargs"] = {}
        width = domain.get("boundary_width", "auto")
        manual = st.checkbox("Specify boundary trimming explicitly", value=width != "auto",
                             key=f"lab_boundary_manual_{source}_{record_name}_{variant}")
        if manual:
            current_widths = ((width if isinstance(width, (list, tuple))
                               else [width] * len(shape)) if width != "auto"
                              else [max(1, n // 10) for n in shape])
            widths = [int(st.number_input(
                f"Trim each side of axis {i}", min_value=0, max_value=max(0,(n-10)//2),
                value=min(int(current_widths[i]), max(0,(n-10)//2)),
                key=f"lab_boundary_{source}_{record_name}_{variant}_{i}"))
                for i, n in enumerate(shape)]
            domain["boundary_width"] = widths[0] if len(widths) == 1 else widths
        else:
            domain["boundary_width"] = "auto"
        a, b, c = st.columns(3)
        native_dp = int(sp.get("data_fun_pow", 3))
        native_derp = int(sp.get("deriv_fun_pow", 2))
        if native_dp > 4 or native_derp > 4:
            st.warning("Native field/derivative powers exceed this public demo's "
                       "limit of 4; the controls show their safely capped values.")
        sp["data_fun_pow"] = int(a.slider(
            "Maximum field power", 1, 4, min(4, max(1, native_dp)),
            key=f"lab_data_power_{source}_{record_name}_{variant}"))
        sp["deriv_fun_pow"] = int(b.slider(
            "Maximum derivative power", 1, 4, min(4, max(1, native_derp)),
            key=f"lab_deriv_power_{source}_{record_name}_{variant}"))
        old_factors = sp.get("equation_factors_max_number", 1)
        allow_products = (max(old_factors.get("factors_num", [1])) >= 2 if
                          isinstance(old_factors, dict) else old_factors >= 2)
        products = c.checkbox("Allow products of 2 factors", value=allow_products,
                              key=f"lab_products_{source}_{record_name}_{variant}")
        if products:
            sp["equation_factors_max_number"] = (
                old_factors if isinstance(old_factors, dict) and allow_products
                else {"factors_num": [1, 2], "probas": [0.65, 0.35]})
        else:
            sp["equation_factors_max_number"] = 1
        st.caption("Existing equation-specific forcing, coordinate and trigonometric "
                   "token families remain intact unless edited in the validated YAML.")
        if len(shape) > 1:
            tokens = list(sp.get("tokens") or [])
            has_inverse = any(x.get("family") == "inverse_space_coordinate" for x in tokens)
            inverse = st.checkbox("Include 1/x token (requires x≠0)",
                                  value=has_inverse,
                                  key=f"lab_inverse_{source}_{record_name}_{variant}")
            if inverse != has_inverse:
                if inverse:
                    tokens.append({"family": "inverse_space_coordinate", "axis": 1})
                else:
                    tokens = [x for x in tokens
                              if x.get("family") != "inverse_space_coordinate"]
                sp["tokens"] = tokens
    if detail == "Advanced":
        st.markdown("**Objectives, sparsity and compute limits**")
    if detail == "Advanced":
        from dataclasses import asdict
        from epde.interface.search_config import (
            METRIC_MENUS, SPARSITY_REGISTRY, load_search_config,
        )
        defaults = asdict(load_search_config(search).objectives)
        a, b = st.columns(2)
        for col, field, title in (
            (a, "discrepancy_metric", "Discrepancy metric"),
            (b, "second_objective", "Second objective"),
            (a, "instability_metric", "Instability estimator"),
        ):
            values = list(METRIC_MENUS[field])
            current = objectives.get(field, defaults[field])
            if len(values) <= 7:
                objectives[field] = col.pills(
                    title, values, default=current, selection_mode="single",
                    key=f"lab_obj_{field}_{source}_{record_name}_{variant}") or current
            else:
                objectives[field] = col.selectbox(
                    title, values, index=values.index(current),
                    key=f"lab_obj_{field}_{source}_{record_name}_{variant}")
        values = list(SPARSITY_REGISTRY)
        current = objectives.get("sparsity_cls", defaults["sparsity_cls"])
        if current not in values:
            current = "vwsr"
        if len(values) <= 7:
            objectives["sparsity_cls"] = b.pills(
                "Sparsity operator", values, default=current, selection_mode="single",
                key=f"lab_sparsity_{source}_{record_name}_{variant}") or current
        else:
            objectives["sparsity_cls"] = b.selectbox(
                "Sparsity operator", values, index=values.index(current),
                key=f"lab_sparsity_{source}_{record_name}_{variant}")
        cache_pct = float(resource.get("memory_for_cache", 2))
        cache_limit = 16.0 if local_full else 4.0
        if not 1 <= cache_pct <= cache_limit:
            st.warning(f"Cache percentage is limited to 1–{cache_limit:g}% "
                       "of this machine's RAM.")
        resource["memory_for_cache"] = float(st.number_input(
            "EPDE tensor cache: % of host RAM", 1.0, cache_limit,
            value=max(1.0, min(cache_limit, cache_pct)),
            step=1.0, key=f"lab_cache_{source}_{record_name}_{variant}"))
        st.caption("Custom Python operator implementations and solver-based "
                   "fitness are not part of this interactive search interface.")

    with st.popover("Import / Export YAML and full configuration"):
        st.caption("The effective resolved scientific configuration is shown below. "
                   "If you import overrides, they apply to these controls as a "
                   "validated declarative patch; no scripts are executed.")
        uploaded = st.file_uploader("Import search-only YAML overrides",
                                    type=["yml", "yaml"],
                                    key=f"lab_yaml_{source}_{record_name}_{variant}")
        if uploaded is not None:
            patch = parse_search_yaml(uploaded.getvalue())["search"]
            if set(patch) - {"domain", "preprocessing", "search_space",
                              "objectives", "evolution", "runtime", "solver"}:
                raise ValueError("Unknown YAML group")
            # Never allow arbitrary imported operator mappings or solver execution.
            forbidden = (patch.get("evolution") or {}).get("operators")
            if forbidden:
                raise ValueError("Custom operators cannot be imported into public Lab")
            if (patch.get("solver") or {}).get("use_solver"):
                raise ValueError("Solvers cannot be enabled in public Lab")
            search = deep_merge(search, patch)
            st.caption("Imported YAML takes precedence; adjust your YAML to change "
                       "fields already selected in the visible controls.")
        search.setdefault("solver", {})["use_solver"] = False
        search["solver"]["device"] = "cpu"
        st.code(yaml.safe_dump({"search": search}, sort_keys=False,
                               allow_unicode=True), language="yaml")
        st.download_button(
            "Export effective YAML", dump_search_yaml({"search": search}),
            file_name=f"{record_name or 'custom'}_{variant}_effective.yaml",
            mime="application/x-yaml", key=f"lab_export_{source}_{record_name}_{variant}")
    search.setdefault("solver", {})["use_solver"] = False
    search["solver"]["device"] = "cpu"
    validate_public_search(search, shape, override=local_full)
    inverse = any(t.get("family") == "inverse_space_coordinate"
                  for t in search.get("search_space", {}).get("tokens", []))
    if inverse:
        import numpy as np
        if built_in:
            x_coordinates = np.asarray(problem.grids[1])
        else:
            x_coordinates = np.asarray(arrays["axis_1"])
        if np.any(np.isclose(x_coordinates, 0, atol=1e-12)):
            raise ValueError("The 1/x token cannot be used on a spatial grid containing x=0")
    if built_in:
        # For both EPDE and PySINDy use the SAME complete search space and
        # differentiator. No other baseline configuration is overwritten.
        # Changes (if any) remain explicit in the recorded run identity.
        overrides = {"search": search} if search != base or variant != "default" else {}
        custom_settings = None
    else:
        overrides = {}
        custom_settings = {"seed": int(seed), "noise": float(noise),
                           "search": search, "label": label, "origin": origin,
                           "effort": preset}
        if reference:
            custom_settings["reference_truth"] = reference
    return variant, overrides, custom_settings, preset
