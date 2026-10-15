"""Equation Discovery Lab: run EPDE, PySINDy (and later an LLM agent) on the same
input and look at the results side by side.

The page never ranks methods: their selection rules and criteria differ. It
shows, for every run, the equation, its candidates, how its solution compares
with the data and, when the law is known, whether it was found.
"""
from __future__ import annotations

import hashlib
import io
import base64
import secrets
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from support.common import dataset_label, load_problem, page, show_figure
from support.discovery_lab import job_queue as queue
from support.discovery_lab import ui
from support.discovery_lab.ode import compile_rhs, generate, reference_law, uses_time
from support.discovery_lab.uploads import parse_upload
from support.equations import equation_latex, system_latex

# EPDE_LOCAL_FULL_RUN belongs to the separate native scientific process.
# Streamlit NEVER loads/runs heavy registry datasets, even when launched locally.

from support.minimal_ui import intro, step, wizard_progress
intro("Run a search", "Choose one dataset, configure the methods, then run or export.")

from support.common import link as safe_page_link
safe_page_link("views/9_How_EPDE_works.py",
               "New to equation discovery? See EPDE and SINDy step by step", "🧭")

# Shared saved session across the three application pages.
from support.discovery_lab.session import current_owner
owner = current_owner()

LLM_READY = bool(os.environ.get("EPDE_LLM_SERVICE_URL"))

# The complete catalogue is derived from the original scientific registry.
# Never hand-maintain a partial list of PDE/ODE records here.
from support.discovery_lab.catalog import grouped_catalog, declared_equations
from support.discovery_lab.execution_policy import is_local_only

EFFORTS = {"quick": "Quick (population 8, 3 epochs)",
           "standard": "Standard (population 16, 5 epochs)",
           "thorough": "Thorough (population 24, 8 epochs)"}
METHOD_LABELS = {"epde": "EPDE", "pysindy": "PySINDy (sparse regression)", "llm": "LLM agent"}
RESERVED = {"x", "t", "sin", "cos", "tanh"}


@st.cache_data(show_spinner=False)
def lab_records():
    return grouped_catalog()


def preview(problem_or_arrays, title=None):
    """A small picture of the data before running anything."""
    import matplotlib.pyplot as plt
    if isinstance(problem_or_arrays, dict):
        variables = {k[4:]: np.asarray(v, dtype=float) for k, v in problem_or_arrays.items()
                     if k.startswith("var_")}
        t = np.asarray(problem_or_arrays["axis_0"], dtype=float)
        x = np.asarray(problem_or_arrays.get("axis_1", []), dtype=float)
    else:
        variables = {k: np.asarray(v, dtype=float) for k, v in problem_or_arrays.data.items()}
        t = np.unique(np.asarray(problem_or_arrays.grids[0]))
        x = np.unique(np.asarray(problem_or_arrays.grids[1])) if problem_or_arrays.dim else np.array([])
    field = next(iter(variables.values()))
    if field.ndim == 1:
        fig, ax = plt.subplots(figsize=(9, 2.4))
        for name, values in list(variables.items())[:5]:
            ax.plot(t, values, lw=1.2, label=name)
        ax.set_xlabel("t")
        ax.legend(fontsize=8)
        ax.grid(alpha=.25)
    else:
        name, values = next(iter(variables.items()))
        while values.ndim > 2:
            values = values[..., values.shape[-1] // 2]
        fig, ax = plt.subplots(figsize=(9, 2.6))
        lim = np.nanpercentile(np.abs(values), 99) or 1.0
        im = ax.imshow(values.T, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim,
                       extent=[t[0], t[-1], x[0], x[-1]] if x.size else None)
        ax.set_xlabel("t")
        ax.set_ylabel("x")
        ax.set_title(name, fontsize=9)
        fig.colorbar(im, ax=ax)
    if title:
        ax.set_title(title, fontsize=9)
    fig.tight_layout()
    show_figure(fig)


def arrays_digest(arrays):
    digest = hashlib.sha256()
    for key in sorted(arrays):
        value = np.ascontiguousarray(arrays[key])
        digest.update(key.encode())
        digest.update(str(value.dtype).encode() + str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()[:16]


@st.cache_data(show_spinner=False)
def example_files():
    """Ready-made files showing the accepted formats (from built-in records)."""
    from epde_bench.datasets import load
    lv = load("lv")
    t = np.unique(np.asarray(lv.grids[0]))
    table = pd.DataFrame({"t": t, **{name: np.asarray(v) for name, v in lv.data.items()}})
    csv = table.to_csv(index=False).encode()
    heat = load("nir1_heat")
    buffer = io.BytesIO()
    np.savez_compressed(buffer, axis_0=np.unique(np.asarray(heat.grids[0])),
                        axis_1=np.unique(np.asarray(heat.grids[1])),
                        axis_names=np.array(["t", "x"]),
                        **{f"var_{k}": np.asarray(v) for k, v in heat.data.items()})
    return csv, buffer.getvalue()



def _restore_wizard_widgets():
    for key, value in st.session_state.get("lab_widgets_saved", {}).items():
        if key not in st.session_state:
            st.session_state[key] = value

def _save_wizard_widgets():
    # Widgets are removed when they are not rendered in a later Streamlit rerun.
    # Mirror their values into permanent state before advancing the wizard.
    names = ("lab_", "portable_", "heavy_", "native_advanced_")
    saved = dict(st.session_state.get("lab_widgets_saved", {}))
    saved.update({
        k: v for k, v in st.session_state.items()
        if k.startswith(names) and k not in (
            "lab_widgets_saved", "lab_stage", "lab_draft_input",
            "lab_draft_config", "lab_started_jobs",
        )
    })
    st.session_state["lab_widgets_saved"] = saved

def _next_stage(target):
    _save_wizard_widgets()
    st.session_state["lab_stage"] = target
    st.rerun()

st.session_state.setdefault("lab_stage", 1)
wizard_stage = st.session_state["lab_stage"]
if wizard_stage not in (1, 2, 3):
    wizard_stage = 1
    st.session_state["lab_stage"] = 1
_restore_wizard_widgets()
wizard_progress(wizard_stage)
if wizard_stage > 1:
    if st.button("← New experiment", key="lab_new_experiment"):
        for key in ("lab_draft_input", "lab_draft_config", "lab_started_jobs", "lab_widgets_saved"):
            st.session_state.pop(key, None)
        st.session_state["lab_stage"] = 1
        st.rerun()

main_col, summary_col = st.columns([3.0, 1.25], gap="medium")
with main_col:
    if wizard_stage == 1:
        step(1, "Choose method and equation")
        options = ["epde", "pysindy"] + (["llm"] if LLM_READY else [])
        st.session_state.setdefault("lab_discovery_methods", ["epde", "pysindy"])
        methods = st.pills(
            "Discovery methods", options,
            key="lab_discovery_methods",
            selection_mode="multi", format_func=METHOD_LABELS.get,
            help="Choose one or both. Each method receives identical input data, "
                 "noise and seed; their optimizers and scoring criteria remain distinct."
        )
        if len(methods) == 2 and set(methods) == {"epde", "pysindy"}:
            st.caption("Both algorithms will be saved as separate experiments in one "
                       "comparison group.")
        if "llm" in methods:
            st.warning("The configured LLM service receives a sample of your data; "
                       "do not send private measurements.")

        st.markdown("**Equation / dataset**")
        source = st.segmented_control("Data source",
                          ["Built-in equation", "Generate an ODE", "Upload your data"],
                          key="lab_input_source",
                          default="Built-in equation",
                          help="Built-in equations have a known law, so the result is checked against it. "
                               "A generated ODE is checked against the law you type. Uploaded data have "
                               "no known law: the result is checked against the data only.")
        arrays, record_name, label, reference, origin = None, None, None, None, None
        noise, seed = 0.0, 0
        space = {}
        is_heavy = False

        if source == "Built-in equation":
            groups = lab_records()
            requested = st.session_state.pop("catalog_requested_record", None)
            default_group = next(
                (group for group, names in groups.items()
                 if requested in names and group != "All registered datasets"),
                "Core benchmark",
            )
            group_labels = {
                "All registered datasets": "All",
                "Ordinary differential equations": "ODE",
                "ODE systems": "Systems",
                "PDE · 1 spatial dimension": "PDE 1D",
                "PDE · 2 spatial dimensions": "PDE 2D",
                "PDE · 3 spatial dimensions": "PDE 3D",
                "Core benchmark": "Core",
                "Extended benchmark": "Extended",
                "Measured data": "Measured",
                "Other data": "Other",
            }
            # Persist Catalog -> Lab selection across Streamlit widget reruns.
            # Without stable keys the first rerun resets the chosen heavy PDE
            # to the first Core benchmark dataset.
            if requested in groups.get(default_group, []):
                st.session_state["lab_builtin_group"] = default_group
                st.session_state["lab_builtin_equation"] = requested
            if st.session_state.get("lab_builtin_group") not in groups:
                st.session_state["lab_builtin_group"] = default_group
            st.caption("Filter the full registry or type an equation name.")
            group = st.pills(
                "Dataset group", list(groups), default=default_group,
                selection_mode="single",
                format_func=lambda x: group_labels.get(x, x),
                key="lab_builtin_group",
            ) or default_group
            names = groups[group]
            if st.session_state.get("lab_builtin_equation") not in names:
                st.session_state["lab_builtin_equation"] = names[0]
            record_name = st.selectbox(
                "Equation (search by name)", names,
                format_func=lambda x: f"{dataset_label(x)} · {x}",
                key="lab_builtin_equation",
            )
            is_heavy = is_local_only(record_name)
            if is_heavy:
                # The original PySINDy adapter cannot represent every PDE.
                # One click removes only unsupported methods; no fake solver fallback.
                from epde_bench.baselines import unsupported_reason
                incompatibility = unsupported_reason(record_name)
                if incompatibility and "pysindy" in methods and "epde" in methods:
                    st.caption("PySINDy is not supported for this equation: " + incompatibility)
                    st.button(
                        "Use EPDE only for this equation",
                        key="heavy_supported_method_" + record_name,
                        on_click=lambda: st.session_state.update(
                            {"lab_discovery_methods": ["epde"]}),
                    )
                # Keep heavy data in the registry without loading it on shared hosting.
                from epde_bench.datasets import REGISTRY
                label = dataset_label(record_name)
                spec = REGISTRY[record_name]
                st.caption("**" + label + "** · " + spec.kind + " · " + spec.suite)
                laws = declared_equations(record_name)
                if laws:
                    st.markdown("**Governing equation(s)**")
                    for law in laws:
                        st.code(law, language=None)
                st.info("This equation requires native scientific computing. "
                           "Choose the complete EPDE/PySINDy settings below, then download "
                           "the repository and an exact experiment configuration. "
                           "No heavy dataset is loaded or searched on this website.")
                st.page_link("views/14_Data_catalog.py",
                             label="Inspect original data and availability")
                a, b = st.columns(2)
                noise = float(a.number_input("Noise, % of standard deviation",
                                             0.0, 10.0, 0.0, 0.5,
                                             key="heavy_noise_" + record_name))
                seed = int(b.number_input("Seed", 0, 999, 0,
                                          key="heavy_seed_" + record_name))
            if not is_heavy:
                from epde_bench.config import load_config
                try:
                    original_profile = load_config(record_name, "default")
                    problem = load_problem(record_name, json.dumps(
                        original_profile.get("loader", {}), sort_keys=True))
                except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
                    st.error("The dataset is registered, but the numerical files or "
                             "its configuration are not available in this checkout: " + str(exc))
                    st.caption("Registration does not guarantee that large or externally "
                               "distributed scientific data files have been downloaded.")
                    st.stop()
                label = dataset_label(record_name)
                st.markdown("**Governing equation**")
                if problem.truth:
                    st.caption("Reference law is only used to evaluate results; it is never given to the discovery algorithms.")
                    st.latex(system_latex(problem.truth, problem.axis_names))
                else:
                    st.caption("No authoritative reference law is declared for this dataset.")
                st.markdown("**Data preview**")
                preview(problem)
                a, b = st.columns(2)
                noise = a.number_input("Noise, % of the standard deviation", 0.0, 10.0, 0.0, 0.5,
                                       key="lab_noise_" + record_name,
                                       help="Gaussian noise added to the data before the search.")
                seed = int(b.number_input("Seed", 0, 999, 0,
                                          key="lab_seed_" + record_name,
                                          help="Fixes the noise and the search."))
                st.caption(f"{problem.kind.replace('_', ' ')}, grid {' × '.join(map(str, problem.shape))}. "
                           "Derivatives, tokens and boundary are the stored settings of this record.")
        elif source == "Generate an ODE":
            st.caption("Type the right-hand side of du/dt = f(t, u). Allowed: t, u, numbers, + − * / **, "
                       "sin, cos, tanh. The expression is parsed, never executed as code.")
            expression = st.text_input("du/dt =", value="-0.5*u + sin(t)")
            a, b, c = st.columns(3)
            horizon = a.number_input("End time", 1.0, 30.0, 10.0)
            u0 = b.number_input("u(0)", -10.0, 10.0, 1.0)
            count = c.number_input("Samples", 60, 3000, 301, step=20)
            try:
                compile_rhs(expression)
                arrays = generate(expression, horizon, u0, int(count))
                law, frequency, note = reference_law(expression)
                label = f"Generated ODE: du/dt = {expression}"
                origin = "generated_ode"
                if law:
                    reference = [law]
                    st.markdown("**Law to recover** (in the form EPDE writes equations)")
                    st.latex(equation_latex(law, ("t",)))
                else:
                    st.warning(f"This law cannot be written with the searched terms ({note}); the "
                               "result is checked against the data only.")
                space = {"time_order": 1, "space_order": None, "frequency": frequency, "power": 3,
                         "time_terms": uses_time(expression)}
                preview(arrays)
            except (ValueError, OverflowError, TypeError) as exc:
                st.error(f"Cannot generate this equation: {exc}")
        else:
            st.markdown("**Supported formats**")
            st.caption("CSV: time column followed by numerical variables. "
                       "NPZ: axis_0, optional axis_1 and var_<name> arrays. "
                       "NPY: time × space array; enter coordinate bounds below. "
                       "Maximum upload: 8 MiB / 300,000 values.")
            csv, npz = example_files()
            c1, c2 = st.columns(2)
            c1.download_button("Example CSV", csv, "lotka_volterra.csv", "text/csv")
            c2.download_button("Example NPZ", npz, "heat_equation.npz",
                               "application/octet-stream")
            uploaded = st.file_uploader("Your file", type=["csv", "npy", "npz"])
            if uploaded is not None:
                try:
                    arrays = parse_upload(uploaded.getvalue(), uploaded.name)
                    if uploaded.name.lower().endswith(".npy"):
                        a, b = st.columns(2)
                        t_end = a.number_input("Time from 0 to", 1e-6, 1e6, 1.0, format="%g")
                        x_end = b.number_input("Space from 0 to", 1e-6, 1e6, 1.0, format="%g")
                        arrays["axis_0"] = np.linspace(0.0, t_end, arrays["var_u"].shape[0])
                        arrays["axis_1"] = np.linspace(0.0, x_end, arrays["var_u"].shape[1])
                    renamed = []
                    for key in [k for k in arrays if k.startswith("var_")]:
                        if key[4:] in RESERVED:
                            arrays[f"var_{key[4:].upper()}"] = arrays.pop(key)
                            renamed.append(f"{key[4:]} → {key[4:].upper()}")
                    if renamed:
                        st.info("Renamed variables that clash with coordinate or function names: "
                                + ", ".join(renamed))
                    label = f"Uploaded: {uploaded.name}"
                    origin = "uploaded"
                    axes_count = len([k for k in arrays if k.startswith("axis_") and k != "axis_names"])
                    variables = [k[4:] for k in arrays if k.startswith("var_")]
                    st.success(f"{len(variables)} variable(s) ({', '.join(variables)}) on "
                               f"{axes_count} axis/axes")
                    preview(arrays)
                    space = {"time_order": 2, "space_order": 3 if axes_count > 1 else None,
                             "frequency": None, "power": 3, "time_terms": True}
                except (ValueError, TypeError, OSError, KeyError) as exc:
                    st.error(f"Cannot use this file: {exc}")
                    arrays = None
            if arrays is not None:
                noise = st.number_input("Extra noise, % of the standard deviation", 0.0, 10.0, 0.0, 0.5,
                                        help="Usually 0 for measured data.")


        input_ready = bool(methods) and (record_name is not None or arrays is not None)
        if st.button("Next: configure methods →", key="lab_next_data",
                     type="primary", disabled=not input_ready,
                     use_container_width=True):
            st.session_state["lab_draft_input"] = dict(
                methods=list(methods), source=source, arrays=arrays,
                record_name=record_name, label=label, reference=reference,
                origin=origin, noise=float(noise), seed=int(seed),
                space=space, is_heavy=is_heavy,
            )
            st.session_state.pop("lab_started_jobs", None)
            st.session_state.pop("lab_draft_config", None)
            st.session_state["lab_confirm_science"] = False
            _next_stage(2)

    else:
        draft = st.session_state.get("lab_draft_input")
        if not draft:
            st.session_state["lab_stage"] = 1
            st.rerun()
        methods = draft["methods"]
        source = draft["source"]
        arrays = draft["arrays"]
        record_name = draft["record_name"]
        label = draft["label"]
        reference = draft["reference"]
        origin = draft["origin"]
        noise = draft["noise"]
        seed = draft["seed"]
        space = draft["space"]
        is_heavy = draft["is_heavy"]
        if record_name and not is_heavy:
            from epde_bench.config import load_config
            original_profile = load_config(record_name, "default")
            problem = load_problem(record_name, json.dumps(
                original_profile.get("loader", {}), sort_keys=True))
        if wizard_stage == 2:
            step(2, "Configure method settings")
            st.caption("Start with the recommended profile. Each method keeps its own scientific settings.")
            from support.discovery_lab.lab_settings import render_settings
            from support.discovery_lab.sindy_settings import render_sindy_settings

            variant, overrides, settings, selected_budget = "default", {}, None, "Preset"
            sindy_settings = None
            settings_valid = True
            heavy_methods = []


            def epde_editor():
                st.markdown("**EPDE · evolutionary search**")
                st.caption("Population, evolution epochs, Pareto objectives, token pool and "
                           "differentiation control EPDE only. None of these optimizer "
                           "parameters is applied to PySINDy.")
                return render_settings(
                    record_name=record_name, problem=problem if record_name else None,
                    arrays=arrays, space=space, source=source, methods=["epde"],
                    noise=noise, seed=seed, label=label, origin=origin, reference=reference,
                )


            def sindy_editor():
                return render_sindy_settings(
                    record_name=record_name, problem=problem if record_name else None,
                    arrays=arrays, space=space, source=source,
                )


            try:
                if is_heavy:
                    from support.discovery_lab.heavy_settings import render_heavy_epde, render_heavy_sindy
                    if "epde" in methods and "pysindy" in methods:
                        epde_tab, sindy_tab = st.tabs(
                            ["🧬 EPDE · Evolution", "📐 PySINDy · Sparse regression"])
                        with epde_tab:
                            heavy_methods.append(render_heavy_epde(record_name))
                        with sindy_tab:
                            candidate = render_heavy_sindy(record_name)
                            if candidate is not None:
                                heavy_methods.append(candidate)
                            else:
                                settings_valid = False
                    elif "epde" in methods:
                        heavy_methods.append(render_heavy_epde(record_name))
                    elif "pysindy" in methods:
                        candidate = render_heavy_sindy(record_name)
                        if candidate is not None:
                            heavy_methods.append(candidate)
                        else:
                            settings_valid = False
                    else:
                        st.info("Select EPDE and/or PySINDy to configure the native experiment.")
                elif "epde" in methods and "pysindy" in methods:
                    epde_tab, sindy_tab = st.tabs(
                        ["🧬 EPDE · Evolution", "📐 PySINDy · Sparse regression"])
                    with epde_tab:
                        variant, overrides, settings, selected_budget = epde_editor()
                    with sindy_tab:
                        sindy_settings = sindy_editor()
                elif "epde" in methods:
                    with st.container(border=True):
                        variant, overrides, settings, selected_budget = epde_editor()
                elif "pysindy" in methods:
                    with st.container(border=True):
                        sindy_settings = sindy_editor()
                elif "llm" in methods:
                    st.caption("The LLM adapter uses the common input shown above.")
                else:
                    st.info("Select EPDE or PySINDy in step 1 to show its scientific controls.")
            except (ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
                st.error(f"Please correct the scientific configuration before running: {exc}")
                settings_valid = False

            budget_code = {
                "Preset": "standard",
                "Quick": "quick",
                "Thorough": "thorough",
                "Custom": "custom",
            }.get(selected_budget, "standard")


            controls = st.columns([1, 2])
            with controls[0]:
                if st.button("← Back to data", key="lab_back_data",
                             use_container_width=True):
                    _next_stage(1)
            with controls[1]:
                if st.button("Next: review configuration →", type="primary",
                             key="lab_next_settings", use_container_width=True,
                             disabled=not settings_valid or not methods
                                 or (is_heavy and len(heavy_methods) != len(methods))
                                 or (not is_heavy and
                                     ("epde" in methods and settings is None and record_name is None
                                      or "pysindy" in methods and sindy_settings is None))):
                    st.session_state["lab_draft_config"] = dict(
                        variant=variant, overrides=overrides, settings=settings,
                        selected_budget=selected_budget, sindy_settings=sindy_settings,
                        heavy_methods=heavy_methods, settings_valid=settings_valid,
                        budget_code=budget_code,
                    )
                    _next_stage(3)

        else:
            config = st.session_state.get("lab_draft_config")
            if not config:
                st.session_state["lab_stage"] = 2
                st.rerun()
            variant = config["variant"]
            overrides = config["overrides"]
            settings = config["settings"]
            selected_budget = config["selected_budget"]
            sindy_settings = config["sindy_settings"]
            heavy_methods = config["heavy_methods"]
            settings_valid = config["settings_valid"]
            budget_code = config["budget_code"]

            st.markdown("**Selected input and execution**")
            st.write("**Dataset:**", label or record_name or source)
            st.write("**Methods:**", ", ".join(METHOD_LABELS.get(m,m) for m in methods))
            st.write("**Noise:**", f"{noise:g} %", " · **Seed:**", seed)
            st.caption("Heavy PDE: configuration/export only. Computation runs via the separate native launcher."
                       if is_heavy else "Hosted search: calculations execute on the Streamlit server.")

            st.markdown("**Complete effective scientific configuration**")
            import yaml
            from epde_bench.config import load_config
            review = {}
            if is_heavy:
                for entry in heavy_methods:
                    base = load_config(record_name, entry["variant"],
                                       overrides=entry.get("overrides") or {})
                    review[entry["method"]] = base
            else:
                if "epde" in methods:
                    review["epde"] = (
                        load_config(record_name, variant, overrides=overrides)
                        if record_name else settings
                    )
                if "pysindy" in methods:
                    review["pysindy"] = (
                        load_config(record_name, "pysindy",
                                    overrides=sindy_settings or {})
                        if record_name else sindy_settings
                    )
            for method in methods:
                if method in review:
                    st.markdown("**" + METHOD_LABELS.get(method,method) + "**")
                    st.code(yaml.safe_dump(review[method], sort_keys=False,
                                           allow_unicode=True), language="yaml")
            st.caption("The full configuration above is used for the run/export. "
                       "EPDE and PySINDy have separate, independent optimizer settings.")
            confirmed = st.checkbox(
                "I have checked the input, the methods and the scientific configuration.",
                key="lab_confirm_science",
            )
            if st.button("← Back to settings", key="lab_back_settings",
                         use_container_width=False):
                _next_stage(2)

            step(3, "Review and run")
            ready = confirmed and settings_valid and bool(methods) and (record_name is not None or arrays is not None)
            ready = ready and ("epde" not in methods or (settings is not None or record_name is not None))
            ready = ready and ("pysindy" not in methods or sindy_settings is not None or is_heavy)
            if is_heavy:
                ready = confirmed and settings_valid and bool(methods) and len(heavy_methods) == len(methods)
            if ready and not is_heavy:
                from support.discovery_lab.execution_policy import classify_epde
                from epde_bench.config import load_config, resolve_for_problem
                try:
                    configurations = []
                    if "epde" in methods:
                        configurations.append(
                            load_config(record_name, variant, overrides=overrides)["search"]
                            if record_name else settings["search"])
                    if "pysindy" in methods:
                        configurations.append(sindy_settings["search"])
                    if not configurations:
                        configurations.append(
                            load_config(record_name, "default")["search"] if record_name
                            else settings["search"])
                    if record_name:
                        shape, nvars = problem.shape, len(problem.variables)
                    else:
                        field_keys = [k for k in arrays if k.startswith("var_")]
                        shape = tuple(np.asarray(arrays[field_keys[0]]).shape)
                        nvars = len(field_keys)
                    for config in configurations:
                        allowed, reason, _ = classify_epde(record_name, shape, nvars, config)
                        if not allowed:
                            ready = False
                            st.warning("The selected dataset or budget exceeds the interactive "
                                       "demo limit: " + reason + ". Reduce the search complexity.")
                            break
                except (ValueError, KeyError, TypeError, IndexError, AttributeError):
                    ready = False
                    st.warning("Please correct the scientific configuration before running.")

            if is_heavy:
                st.markdown("**This equation runs on your computer**")
                st.caption("The ZIP contains the complete selected method configurations, "
                           "dataset name, noise, seed, scientific source fingerprint, "
                           "and a manifest checksum. No numerical data are transferred.")
                if ready:
                    try:
                        from support.discovery_lab.portable import make_manifest, bundle_bytes
                        manifest = make_manifest(record_name, float(noise), int(seed), heavy_methods)
                        st.download_button(
                            "Download experiment.zip",
                            data=bundle_bytes(manifest),
                            file_name="experiment.zip",
                            mime="application/zip",
                            type="primary",
                            key="heavy_export_" + record_name,
                        )
                    except (ValueError, TypeError, OSError) as exc:
                        st.error("Could not prepare reproducible experiment: " + str(exc))
                else:
                    st.caption("Confirm the configuration above to unlock the download."
                               if not confirmed else "Correct the unsupported method or settings in step 2.")
                st.link_button(
                    "Download EPDE repository ZIP",
                    "https://github.com/MakSoS1/EPDE/archive/refs/heads/demo/streamlit-community-cloud.zip")
                st.markdown("**Run the downloaded experiment locally**")
                with st.container():
                    st.write("Extract the repository and copy **experiment.zip** into its root folder.")
                    windows, unix = st.tabs(["Windows", "macOS / Linux"])
                    with windows:
                        st.write("Double-click **run-portable.cmd** in the extracted folder.")
                        st.code("run-portable.cmd", language="bat")
                    with unix:
                        st.write("Open Terminal in the extracted folder and run:")
                        st.code("bash run-portable.sh", language="bash")
                    st.caption("The first run installs pinned dependencies. Internet access is required.")
                st.caption("When finished, upload **experiment-results.zip** on the Results page.")
                st.page_link("views/11_Discovery_results.py", label="Upload or view results", icon=":material/upload_file:")
            elif st.button("Start search: " + " & ".join(METHOD_LABELS[m] for m in methods)
                         if methods else "Choose a method",
                         type="primary", disabled=not ready or bool(st.session_state.get("lab_started_jobs")), key="lab_start"):
                try:
                    if record_name:
                        key = f"record|{record_name}|{noise:g}|{seed}"
                        requests = []
                        for m in methods:
                            if m == "epde":
                                method_overrides = overrides
                            elif m == "pysindy":
                                method_overrides = sindy_settings
                            else:
                                method_overrides = {}
                            requests.append({
                                "method": m, "source": "record", "dataset": record_name,
                                "variant": "pysindy" if m == "pysindy" else
                                           variant if m == "epde" else "default",
                                "overrides": method_overrides,
                                "noise": float(noise), "seed": seed,
                                "effort": budget_code if m == "epde" else "standard",
                                "display": {"label": label, "input_key": key},
                            })
                    else:
                        key = f"upload|{arrays_digest(arrays)}|{noise:g}|{seed}"
                        from support.discovery_lab.lab_settings import _custom_base
                        fallback = _custom_base(len([
                            k for k in arrays if k.startswith("axis_") and k != "axis_names"
                        ]) - 1, generated=source == "Generate an ODE")
                        shared_settings = dict(settings or {
                            "search": sindy_settings["search"] if sindy_settings else fallback,
                            "seed": int(seed), "noise": float(noise),
                            "label": label, "origin": origin,
                        })
                        shared_settings["method_configs"] = {}
                        if settings is not None and "epde" in methods:
                            shared_settings["method_configs"]["epde"] = {
                                "search": settings["search"]}
                        if sindy_settings is not None and "pysindy" in methods:
                            shared_settings["method_configs"]["pysindy"] = sindy_settings
                        if reference:
                            shared_settings["reference_truth"] = reference
                        data_path, config_path = queue.save_upload(owner, arrays, shared_settings)
                        requests = [{
                            "method": m, "source": "upload", "dataset": "custom",
                            "noise": float(noise), "seed": seed,
                            "effort": budget_code if m == "epde" else "standard",
                            "data_path": str(data_path), "config_path": str(config_path),
                            "display": {"label": label, "input_key": key},
                        } for m in methods]
                    # A joint capacity check avoids partially submitting paired runs.
                    for m in methods:
                        queue.check_capacity(owner, m, count=len(methods))
                    ids = [queue.submit(owner, req) for req in requests]
                    st.session_state["discovery_selected_job"] = ids[0]
                    st.session_state["discovery_compare_key"] = key
                    st.session_state["lab_started_jobs"] = ids
                    st.success(f"{len(ids)} search(es) started for the same input. "
                               "Open Results to track progress or Compare to view the equations.")
                    st.page_link("views/11_Discovery_results.py", label="See results", icon="🗂️")
                    if len(ids) > 1:
                        st.page_link("views/12_Discovery_compare.py", label="Compare methods", icon="⚖️")
                except (ValueError, OSError, TypeError) as exc:
                    st.error(f"Cannot submit the search: {exc}")



            if st.session_state.get("lab_started_jobs"):
                @st.fragment(run_every=4)
                def _running_searches():
                    wanted=set(st.session_state.get("lab_started_jobs") or [])
                    matching=[r for r in queue.list_jobs(owner) if r["id"] in wanted]
                    done=sum(r.get("state") in ("finished","error","timeout","interrupted","stopped")
                             for r in matching)
                    st.progress(done / len(wanted) if wanted else 0,
                                text=f"{done} of {len(wanted)} experiment(s) completed")
                    for row in matching:
                        st.write("**"+str(METHOD_LABELS.get(row.get("method"),row.get("method")))+"**",
                                 "—",row.get("state","unknown"))
                    if wanted and done==len(wanted):
                        st.success("Search completed. Open Results to inspect equations, "
                                   "or Compare for matching input runs.")
                    st.page_link("views/11_Discovery_results.py",
                                 label="View results", icon=":material/insights:")
                    if len(wanted)>1:
                        st.page_link("views/12_Discovery_compare.py",
                                     label="Compare experiments", icon=":material/compare_arrows:")
                _running_searches()


with summary_col:
    with st.container(border=True):
        st.markdown("**Experiment summary**")
        if record_name:
            st.markdown("**" + str(label or dataset_label(record_name)) + "**")
        else:
            st.markdown("**" + str(label or source) + "**")
        st.caption("Methods: " + ", ".join(
            METHOD_LABELS.get(method, method) for method in methods) if methods else
            "Choose a method to continue.")
        st.write("Noise:", f"{noise:g}%" if isinstance(noise, (int, float)) else "—")
        st.write("Seed:", seed)
        st.divider()
        if is_heavy:
            st.markdown("**Local execution required**")
            st.caption("Set your parameters here, download the experiment and run the original scientific code on your PC.")
        else:
            st.markdown("**Runs on shared server**")
            st.caption("Compute EPDE and PySINDy on the hosted server within its safety limits.")
        if record_name and "pysindy" in methods:
            from epde_bench.baselines import unsupported_reason
            reason = unsupported_reason(record_name)
            if reason:
                st.caption("PySINDy unavailable for this equation: " + reason)
    st.page_link("views/14_Data_catalog.py", label="Browse all datasets", icon=":material/database:")
