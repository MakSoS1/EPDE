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

page("Equation Discovery Lab", "🧪")
LOCAL_FULL = os.environ.get("EPDE_LOCAL_FULL_RUN") == "1"
if LOCAL_FULL:
    st.success("Full local EPDE environment: the original engine and entire "
               "scientific catalogue are available on this machine. "
               "Load very large data only when needed.")
else:
    st.caption("Light EPDE runs on this free shared server; heavy PDEs are "
               "listed but must be computed locally. PySINDy runs in your browser.")
st.page_link("views/13_Browser_sindy.py", label="⚡ Open PySINDy — compute on my CPU", icon=None)
st.page_link("views/14_Data_catalog.py", label="📚 Browse all ODE/PDE datasets", icon=None)

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


st.subheader("1 · Choose data")
source = st.segmented_control("Data source",
                  ["Built-in equation", "Generate an ODE", "Upload your data"],
                  default="Built-in equation",
                  help="Built-in equations have a known law, so the result is checked against it. "
                       "A generated ODE is checked against the law you type. Uploaded data have "
                       "no known law: the result is checked against the data only.")
arrays, record_name, label, reference, origin = None, None, None, None, None
noise, seed = 0.0, 0
space = {}

if source == "Built-in equation":
    groups = lab_records()
    requested = st.session_state.pop("catalog_requested_record", None)
    default_group = next(
        (group for group, names in groups.items()
         if requested in names and group != "All registered datasets"),
        "All registered datasets",
    )
    group = st.selectbox("Group", list(groups), index=list(groups).index(default_group))
    names = groups[group]
    record_name = st.selectbox("Equation", names,
                               index=names.index(requested) if requested in names else 0,
                               format_func=dataset_label)
    local_full = LOCAL_FULL
    if is_local_only(record_name) and not local_full:
        from epde_bench.datasets import REGISTRY
        spec = REGISTRY[record_name]
        label = dataset_label(record_name)
        st.info("**Full database record:** " + label)
        st.caption("Scientific class: " + spec.kind + " · Dataset suite: " +
                   spec.suite + " · Source: " + spec.source)
        st.code(spec.files, language=None)
        laws = declared_equations(record_name)
        if laws:
            st.markdown("**Declared governing equation(s)**")
            for law in laws:
                st.code(law, language=None)
        else:
            st.caption("This record's law must be read from its original "
                       "dataset loader; it may be data-dependent or unknown.")
        st.warning("This full-resolution dataset is kept in the catalogue, "
                   "but it must be executed on a local scientific environment. "
                   "The free shared server will not load it.")
        st.markdown("**Run the entire EPDE interface on your computer:**")
        st.code(
            "git clone --branch demo/streamlit-community-cloud https://github.com/MakSoS1/EPDE.git\n"
            "cd EPDE\n"
            "uv run --project projects/pic --locked --extra cpu --extra app "
            "streamlit run projects/pic/app/Home.py",
            language="bash")
        st.page_link("views/14_Data_catalog.py", label="View the complete scientific catalogue")
        st.caption("Alternatively open the repository in GitHub Codespaces to "
                   "run the interface through any desktop or mobile browser. "
                   "Full scientific data availability depends on the checkout.")
        st.stop()
    if is_local_only(record_name) and local_full:
        permission_key = "lab_load_heavy_" + record_name
        if not st.session_state.get(permission_key):
            st.warning("This is a large dataset. Loading it may consume substantial RAM "
                       "and CPU even on your own device.")
            if st.button("Load original dataset on this device", type="primary"):
                st.session_state[permission_key] = True
                st.rerun()
            st.stop()
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
    if problem.truth:
        st.markdown("**Known law** — used only to evaluate the result, never as input to discovery")
        st.latex(system_latex(problem.truth, problem.axis_names))
    else:
        st.caption("No authoritative reference equation is declared for this record. "
                   "Reconstruction is evaluated against its numerical data only.")
    if not is_local_only(record_name):
        preview(problem)
    else:
        st.caption("Large local dataset preview is disabled by default to preserve memory.")
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

st.subheader("2 · Compute method")
st.markdown("**EPDE** uses the original scientific engine on the demo server "
            "for light jobs. **PySINDy** runs in your browser from the separate "
            "tab above and does not use the server CPU.")
options = ["epde"] + (["llm"] if LLM_READY else [])
methods = st.pills("Methods", options, default=["epde"],
                    selection_mode="multi", format_func=METHOD_LABELS.get)
if not LLM_READY:
    st.caption("The LLM agent slot is ready but no service is configured on this server.")
elif "llm" in methods:
    st.warning("The LLM service receives a sample of these data. Do not send private measurements.")

st.subheader("3 · Scientific settings")
st.caption("The same scientific settings are used locally and on the server.")
from support.discovery_lab.lab_settings import render_settings

variant, overrides, settings, selected_budget = "default", {}, None, "Preset"
settings_valid = True
try:
    variant, overrides, settings, selected_budget = render_settings(
        record_name=record_name, problem=problem if record_name else None,
        arrays=arrays, space=space, source=source, methods=methods,
        noise=noise, seed=seed, label=label, origin=origin, reference=reference,
    )
except (ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
    st.error(f"Please correct the scientific configuration before running: {exc}")
    settings_valid = False

budget_code = {
    "Preset": "standard",
    "Quick": "quick",
    "Thorough": "thorough",
    "Custom": "custom",
}.get(selected_budget, "standard")

st.subheader("4 · Compute")
SERVER_MODE = "This app · local CPU" if LOCAL_FULL else "Demo server"
location = st.segmented_control("Execution",
    [SERVER_MODE, "My computer"], default=SERVER_MODE,
    key="lab_execution_location",
    help="This app uses the original EPDE engine. On Community Cloud there "
         "are hard CPU limits; a locally launched app runs the full scientific "
         "configuration. My computer uses an independently approved worker.")
if location == "My computer":
    st.caption("Native EPDE and PySINDy run on your computer. The demo site is hosted for free.")
    with st.popover("Install and start local worker"):
        st.code(
            "git clone --branch demo/streamlit-community-cloud https://github.com/MakSoS1/EPDE.git\n"
            "cd EPDE/projects/pic\n"
            "uv sync --locked --extra cpu --extra app\n"
            "uv run --locked --extra cpu --extra app python app/local_worker_service.py",
            language="bash")
        st.caption("Install uv once: https://docs.astral.sh/uv/getting-started/installation/ "
                   "Start this worker yourself and keep its terminal open. "
                   "The browser may ask for permission to access a local service.")
    access = st.text_input("Local worker access code", type="password", key="lab_worker_code")
    if "llm" in methods:
        st.info("Local execution currently supports EPDE and PySINDy, not the private LLM service.")
else:
    access = ""
    if LOCAL_FULL:
        st.caption("Original EPDE computes on this machine; the free Cloud limits "
                   "do not apply. Very large datasets may exhaust local memory.")
    else:
        st.caption("Only small inputs and bounded search budgets may use the free CPU. "
                   "Bigger data and Navier–Stokes always run locally.")

# Show the decision before the user clicks Run; the worker verifies it again.
hosted_allowed, hosted_reason, hosted_diag = True, "", {}
if settings_valid and "epde" in methods:
    from support.discovery_lab.execution_policy import classify_epde
    from epde_bench.config import load_config, resolve_for_problem
    try:
        if record_name:
            effective = load_config(record_name, variant, overrides=overrides)
            resolved = resolve_for_problem(effective, problem)
            shape, nvars = problem.shape, len(problem.variables)
        elif arrays is not None and settings is not None:
            resolved = settings["search"]
            field_keys = [k for k in arrays if k.startswith("var_")]
            shape = tuple(np.asarray(arrays[field_keys[0]]).shape)
            nvars = len(field_keys)
        else:
            resolved, shape, nvars = {}, (), 0
        hosted_allowed, hosted_reason, hosted_diag = classify_epde(
            record_name, shape, nvars, resolved)
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        hosted_allowed, hosted_reason = False, "Invalid scientific settings."
if location == SERVER_MODE and not LOCAL_FULL:
    if hosted_allowed:
        if "epde" in methods:
            st.caption("Server admission: " + hosted_reason)
    else:
        st.warning("This configuration must run locally: " + hosted_reason)
        st.caption("Choose My computer for the original EPDE search or reduce "
                   "the dataset and search budget. Heavy jobs are never queued.")
ready = settings_valid and bool(methods) and (record_name is not None or arrays is not None)
if location == SERVER_MODE and not LOCAL_FULL and "epde" in methods:
    ready = ready and hosted_allowed
if location == "My computer":
    ready = ready and bool(access.strip()) and all(m in ("epde","pysindy") for m in methods)
if st.button("Prepare local runs" if location == "My computer" else
             "Run EPDE here" if LOCAL_FULL else "Run on demo server",
             type="primary", disabled=not ready, key="lab_start"):
    try:
        if record_name:
            key = f"record|{record_name}|{noise:g}|{seed}"
            requests = [{"method": m, "source": "record", "dataset": record_name,
                "variant": ("pysindy" if m == "pysindy" else variant if m == "epde" else "default"),
                "overrides": overrides if m != "llm" else {},
                "noise": float(noise), "seed": seed, "effort": budget_code,
                "display": {"label": label, "input_key": key}} for m in methods]
        else:
            key = f"upload|{arrays_digest(arrays)}|{noise:g}"
            requests = [{"method": m, "source": "upload", "dataset": "custom",
                "noise": float(noise), "seed": seed, "effort": budget_code,
                "display": {"label": label, "input_key": key}} for m in methods]
            if location == "My computer":
                buffer = io.BytesIO()
                np.savez_compressed(buffer, **arrays)
                b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
                if len(b64) > 4_000_000:
                    raise ValueError("Local transfer exceeds 3 MiB")
                for request in requests:
                    request["data_b64"] = b64
                    request["settings"] = settings
            else:
                data_path, config_path = queue.save_upload(owner, arrays, settings)
                for request in requests:
                    request["data_path"] = str(data_path)
                    request["config_path"] = str(config_path)
        if location == "My computer":
            st.session_state["lab_local_pending"] = [
                {"id": secrets.token_hex(12), "request": req} for req in requests]
            st.success("Ready. Start the first local search below.")
        else:
            for m in methods:
                queue.check_capacity(owner, m, count=len(methods))
            ids = [queue.submit(owner, req) for req in requests]
            st.session_state["discovery_selected_job"] = ids[0]
            st.session_state["discovery_compare_key"] = key
            st.success("Server searches queued. Check Results.")
    except (ValueError, OSError, TypeError) as exc:
        st.error(f"Cannot submit the search: {exc}")

pending = st.session_state.get("lab_local_pending") or []
if pending:
    active = pending[0]
    st.markdown("**Run on your computer**")
    st.caption(f"{len(pending)} job(s) remaining. "
               f"Next: {METHOD_LABELS.get(active['request']['method'], active['request']['method'])}")
    if st.button("Discard prepared local jobs", key="discard_local"):
        st.session_state["lab_local_pending"] = []
        st.rerun()
    if not access.strip():
        st.warning("Enter the local worker code to continue.")
    else:
        from support.discovery_lab.local_browser import local_bridge
        output = local_bridge(job=active["request"], token=access.strip(),
                              key="local_job_" + active["id"])
        if output.result is not None:
            try:
                result_id = queue.save_client_result(owner, active["request"], output.result)
                st.session_state["discovery_selected_job"] = result_id
                st.session_state["discovery_compare_key"] = (
                    active["request"].get("display") or {}).get("input_key")
                st.session_state["lab_local_pending"] = pending[1:]
                st.toast("Saved local result in Results", icon="✅")
                st.rerun()
            except (ValueError, OSError, TypeError) as exc:
                st.error(f"Cannot store local result: {exc}")
        if output.failed:
            st.warning("Worker reported a failure. Check the local terminal and retry.")

st.divider()
st.subheader("Your latest experiments")
recent = queue.list_jobs(owner)
if recent:
    active = [r for r in recent if r.get("state") in ("queued", "running")]
    if active:
        st.info(f"{len(active)} experiment(s) running or waiting. "
                "You can navigate away while jobs run on the server.")
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
