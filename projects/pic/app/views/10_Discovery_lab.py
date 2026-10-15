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

# Internal safety flag; never exposed as a mode picker to visitors.
LOCAL_FULL = os.environ.get("EPDE_LOCAL_FULL_RUN") == "1"

from support.minimal_ui import intro, step
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


main_col, summary_col = st.columns([3.45, 1.1], gap="large")
with main_col:
    step(1, "Choose method and equation")
    options = ["epde", "pysindy"] + (["llm"] if LLM_READY else [])
    methods = st.pills(
        "Discovery methods", options, default=["epde", "pysindy"],
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
        with st.expander("Filter dataset collection", expanded=False):
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
        local_full = LOCAL_FULL
        is_heavy = is_local_only(record_name) and not local_full
        if is_heavy:
            # Keep heavy data in the registry without loading it on shared hosting.
            from epde_bench.datasets import REGISTRY
            label = dataset_label(record_name)
            spec = REGISTRY[record_name]
            st.caption("**" + label + "** · " + spec.kind + " · " + spec.suite)
            laws = declared_equations(record_name)
            if laws:
                with st.expander("Original governing equations", expanded=False):
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
        if is_local_only(record_name) and local_full:
            permission_key = "lab_load_heavy_" + record_name
            if not st.session_state.get(permission_key):
                st.warning("This is a large dataset. Loading it may consume substantial RAM "
                           "and CPU even on your own device.")
                if st.button("Load original dataset on this device", type="primary"):
                    st.session_state[permission_key] = True
                    st.rerun()
                st.stop()
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
            with st.expander("Preview data and reference law", expanded=False):
                if problem.truth:
                    st.caption("Known law is used only for evaluation, never to generate candidates.")
                    st.latex(system_latex(problem.truth, problem.axis_names))
                else:
                    st.caption("No authoritative reference equation is available for this record.")
                if not is_local_only(record_name):
                    preview(problem)
                else:
                    st.caption("Large dataset preview is disabled to preserve memory.")
            a, b = st.columns(2)
            noise = a.number_input("Noise, % of the standard deviation", 0.0, 10.0, 0.0, 0.5,
                                   help="Gaussian noise added to the data before the search.")
            seed = int(b.number_input("Seed", 0, 999, 0, help="Fixes the noise and the search."))
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
        with st.expander("Accepted formats and examples", expanded=False):
            st.markdown(
                "- **CSV**: a table, first column time, other columns variables (one ODE or a system). "
                "Example: Lotka–Volterra.\n"
                "- **NPZ**: `axis_0` (time), `axis_1` … (space, 1-D arrays), optional `axis_names`, "
                "and one array `var_<name>` per variable on the grid (time × space). "
                "Example: heat equation.\n"
                "- **NPY**: one 2-D array (time × space); give the coordinate ranges below.\n\n"
                "Up to 8 MiB and 300 000 numbers. Coordinates must be strictly increasing.")
            csv, npz = example_files()
            c1, c2 = st.columns(2)
            c1.download_button("Example CSV (Lotka–Volterra)", csv, "lotka_volterra.csv", "text/csv")
            c2.download_button("Example NPZ (heat equation)", npz, "heat_equation.npz",
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

    step(3, "Run or export")
    ready = settings_valid and bool(methods) and (record_name is not None or arrays is not None)
    ready = ready and ("epde" not in methods or (settings is not None or record_name is not None))
    ready = ready and ("pysindy" not in methods or sindy_settings is not None or is_heavy)
    if is_heavy:
        ready = settings_valid and bool(methods) and len(heavy_methods) == len(methods)
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
                if not allowed and not LOCAL_FULL:
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
                    "1 · Download experiment.zip",
                    data=bundle_bytes(manifest),
                    file_name="experiment.zip",
                    mime="application/zip",
                    type="primary",
                    key="heavy_export_" + record_name,
                )
            except (ValueError, TypeError, OSError) as exc:
                st.error("Could not prepare reproducible experiment: " + str(exc))
        else:
            st.warning("Choose supported methods and correct their settings to export.")
        st.link_button(
            "2 · Download the original EPDE repository (ZIP)",
            "https://github.com/MakSoS1/EPDE/archive/refs/heads/demo/streamlit-community-cloud.zip")
        with st.expander("3 · How to start the downloaded experiment", expanded=True):
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
    elif st.button("Run " + " & ".join(METHOD_LABELS[m] for m in methods)
                 if methods else "Choose a method",
                 type="primary", disabled=not ready, key="lab_start"):
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
            st.success(f"{len(ids)} search(es) started for the same input. "
                       "Open Results to track progress or Compare to view the equations.")
            st.page_link("views/11_Discovery_results.py", label="See results", icon="🗂️")
            if len(ids) > 1:
                st.page_link("views/12_Discovery_compare.py", label="Compare methods", icon="⚖️")
        except (ValueError, OSError, TypeError) as exc:
            st.error(f"Cannot submit the search: {exc}")

    with st.expander("Recent experiments", expanded=False):

        recent = queue.list_jobs(owner)
        if recent:
            active = [r for r in recent if r.get("state") in ("queued", "running")]
            if active:
                st.info(f"{len(active)} experiment(s) running or waiting. "
                        "You can navigate away while the searches are running.")
            for item in recent[:3]:
                state = item.get("state", "unknown")
                st.write(f"**{item.get('label') or item.get('dataset')}** · "
                         f"{METHOD_LABELS.get(item.get('method'), item.get('method'))} · {state}")
            a, b = st.columns(2)
            with a:
                st.page_link("views/11_Discovery_results.py", label="See all Results", icon="🗂️")
            with b:
                st.page_link("views/12_Discovery_compare.py", label="Compare experiments", icon="⚖️")
        else:
            st.caption("Your first experiment will appear here. Later, open Results or Compare "
                       "from the left sidebar.")

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
