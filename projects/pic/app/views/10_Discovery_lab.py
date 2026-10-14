"""Equation Discovery Lab: run EPDE, PySINDy (and later an LLM agent) on the same
input and look at the results side by side.

The page never ranks methods: their selection rules and criteria differ. It
shows, for every run, the equation, its candidates, how its solution compares
with the data and, when the law is known, whether it was found.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import time

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
st.caption("Choose data, run one or several equation-discovery methods on exactly the same input, "
           "then compare what each one found. Methods are compared side by side, not ranked.")
from support.common import link as safe_page_link
safe_page_link("views/9_How_EPDE_works.py",
               "New to equation discovery? See EPDE and SINDy step by step", "🧭")

# Shared saved session across the three application pages.
from support.discovery_lab.session import current_owner
owner = current_owner()

LLM_READY = bool(os.environ.get("EPDE_LLM_SERVICE_URL"))

# Records light enough for an interactive demo, grouped for the selector.
RECORD_GROUPS = {
    "Ordinary differential equations": ["ode", "vdp", "duffing", "nir1_decay", "nir1_logistic",
                                        "nir1_quadratic", "nir1_cubic", "nir1_harmonic", "pend_single"],
    "Systems of ODEs": ["lv", "lorenz"],
    "Partial differential equations": ["burgers", "burgers_inviscid", "ac", "wave", "kdv_cossin",
                                       "pde_compound", "pde_divide"],
    "Partial differential equations, NIR-1 test set": [
        "nir1_advection", "nir1_heat", "nir1_reactiondiff", "nir1_wave", "nir1_advdiff",
        "nir1_h11_gaussian_heat", "nir1_h11_advection_decay", "nir1_h11_viscous_burgers",
        "nir1_h11_klein_gordon", "nir1_h11_biharmonic", "nir1_h11_airy", "nir1_h11_shear",
        "nir1_h11_logistic_field", "nir1_h11_gauss_reactdiff", "nir1_h14_damped_wave",
        "nir1_h14_cole_hopf", "nir1_h14_kpp_front", "nir1_h14_linear_ks", "nir1_h14_beam",
        "nir1_h14_airy_transport", "nir1_h14_cubic_saturation", "nir1_h14_driven_advection",
        "nir1_h14_variable_diffusivity", "nir1_h14_liouville_growth",
        "nir1_h14_hyperdiffusion_growth"],
}
EFFORTS = {"quick": "Quick (population 8, 3 epochs)",
           "standard": "Standard (population 16, 5 epochs)",
           "thorough": "Thorough (population 24, 8 epochs)"}
METHOD_LABELS = {"epde": "EPDE", "pysindy": "PySINDy (sparse regression)", "llm": "LLM agent"}
RESERVED = {"x", "t", "sin", "cos", "tanh"}


@st.cache_data(show_spinner=False)
def lab_records():
    from epde_bench.datasets import REGISTRY
    return {group: [n for n in names if n in REGISTRY] for group, names in RECORD_GROUPS.items()}


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
source = st.radio("Where do the data come from?",
                  ["Built-in equation", "Generate an ODE", "Upload your data"], horizontal=True,
                  help="Built-in equations have a known law, so the result is checked against it. "
                       "A generated ODE is checked against the law you type. Uploaded data have "
                       "no known law: the result is checked against the data only.")
arrays, record_name, label, reference, origin = None, None, None, None, None
noise, seed = 0.0, 0
space = {}

if source == "Built-in equation":
    groups = lab_records()
    group = st.selectbox("Group", list(groups))
    record_name = st.selectbox("Equation", groups[group], format_func=dataset_label)
    from epde_bench.config import load_config
    original_profile = load_config(record_name, "default")
    problem = load_problem(record_name, json.dumps(original_profile.get("loader", {}), sort_keys=True))
    label = dataset_label(record_name)
    st.markdown("**Known law** — used only to check the result, the methods do not see it")
    st.latex(system_latex(problem.truth, problem.axis_names))
    preview(problem)
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

st.subheader("2 · Choose methods")
options = ["epde", "pysindy"] + (["llm"] if LLM_READY else [])
methods = st.multiselect("Run on the same input", options, default=["epde", "pysindy"],
                         format_func=METHOD_LABELS.get)
if not LLM_READY:
    st.caption("The LLM agent slot is ready but no service is configured on this server.")
elif "llm" in methods:
    st.warning("The LLM service receives a sample of these data. Do not send private measurements.")

st.subheader("3 · Scientific settings")
st.caption("Execution currently runs on the Streamlit server. Local execution on your own "
           "CPU/GPU will require an explicitly installed local worker; a web page cannot "
           "silently launch system Python on your computer.")
from support.discovery_lab.lab_settings import render_settings

variant, overrides, settings, selected_budget = "default", {}, None, "Stored / recommended starting values"
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
    "Stored / recommended starting values": "standard",
    "Quick exploratory run": "quick",
    "Thorough exploratory run": "thorough",
    "Exact values": "custom",
}[selected_budget]

ready = settings_valid and (record_name is not None or arrays is not None) and bool(methods)
if st.button(f"Run {len(methods)} search(es) on this input" if methods else "Choose a method",
             type="primary", disabled=not ready):
    try:
        for method in methods:
            queue.check_capacity(owner, method, count=len(methods))
        if record_name:
            key = f"record|{record_name}|{noise:g}|{seed}"
            requests = [{"method": m, "source": "record", "dataset": record_name,
                         "variant": ("pysindy" if m == "pysindy" else variant if m == "epde" else "default"),
                         "overrides": overrides if m != "llm" else {},
                         "noise": float(noise), "seed": seed, "effort": budget_code,
                         "display": {"label": label, "input_key": key}} for m in methods]
        else:
            key = f"upload|{arrays_digest(arrays)}|{noise:g}"
            data_path, config_path = queue.save_upload(owner, arrays, settings)
            requests = [{"method": m, "source": "upload", "dataset": "custom",
                         "data_path": str(data_path), "config_path": str(config_path),
                         "noise": float(noise), "seed": seed, "effort": budget_code,
                         "display": {"label": label, "input_key": key}} for m in methods]
        ids = [queue.submit(owner, request) for request in requests]
        st.session_state["discovery_selected_job"] = ids[0]
        st.session_state["discovery_compare_key"] = key
        st.success(f"Started {len(ids)} search(es). Open Results in the sidebar to follow progress; "
                   "Compare will show completed methods side by side.")
    except (ValueError, OSError) as exc:
        st.error(f"Cannot start: {exc}")


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
