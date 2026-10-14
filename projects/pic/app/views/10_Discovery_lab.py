"""Safe, independent method launcher for the EPDE Demo Track.

It deliberately does not produce a cross-framework leaderboard: objectives and
selection rules differ. The original research pages remain intact locally.
"""
from __future__ import annotations
import json
import os
import time

import numpy as np
import streamlit as st

from support.common import dataset_label, dataset_names, page, show_figure
from support.discovery_lab import queue
from support.discovery_lab.ode import generate, compile_rhs
from support.discovery_lab.uploads import parse_upload

page("Equation Discovery Lab", "🧪")
st.caption("Run an equation-discovery method on a known dataset, your numerical data, or "
           "a safely generated ODE. Methods are selectable, not ranked against one another.")

if "discovery_lab_owner" not in st.session_state:
    st.session_state["discovery_lab_owner"] = queue.new_owner()
with st.expander("Private session / restore jobs after a hard page reload"):
    st.caption("Jobs survive page reloads on the server, but Streamlit may start a new "
               "browser session. This 192-bit recovery code is the only way to reopen "
               "your results without authentication. Keep it private.")
    st.code(st.session_state["discovery_lab_owner"], language=None)
    recovery = st.text_input("Recover a previous session by its secret code", type="password")
    if st.button("Recover session"):
        try:
            queue.owner_path(recovery)
            if not queue.owner_path(recovery).exists():
                raise ValueError("No stored session with that code")
            st.session_state["discovery_lab_owner"] = recovery
            st.rerun()
        except ValueError:
            st.error("Invalid or unknown session recovery code")
owner = st.session_state["discovery_lab_owner"]

source = st.radio("Input", ["Built-in record", "Upload data", "Define an ODE"], horizontal=True)
method_label = st.radio("Algorithm", ["EPDE", "PySINDy", "LLM agent (integration slot)"], horizontal=True)
method = {"EPDE": "epde", "PySINDy": "pysindy", "LLM agent (integration slot)": "llm"}[method_label]
paired = st.checkbox("Run an EPDE + PySINDy pair on identical input (no algorithm ranking)",
                     value=False, disabled=(method == "llm"),
                     help="Uses EPDE/default and PySINDy/pysindy on the same data, "
                          "noise, seed and derivative estimator. Reports each method separately.")
if paired:
    st.caption("Paired discovery compares recovered structures on the same inputs, not "
               "numerical objective values or a cross-method leaderboard. "
               "On built-in records the fixed EPDE/default and PySINDy/pysindy settings are used.")
remote_ready = bool(os.environ.get("EPDE_LLM_SERVICE_URL"))
if method == "llm":
    if not remote_ready:
        st.info("The LLM adapter contract is ready, but Yaroslav's private service is not configured. "
                "No third-party model is invoked in this build.")
    else:
        st.warning("An external LLM service receives a bounded numerical sample of your data. "
                   "Do not use private or sensitive measurements without permission.")

arrays = None
settings = None
chosen_record = None
upload_name = None
expression = None
noise = 0.0
seed = 0

if source == "Built-in record":
    # Lightweight demo set: no Navier-Stokes, expensive multi-dimensional fields,
    # or records with missing files. The registry remains the source of truth.
    supported = {"ode", "vdp", "duffing", "lorenz", "lotka", "burgers", "wave", "ac", "kdv"}
    available = [n for n in dataset_names() if n in supported]
    if not available:
        st.error("No lightweight benchmark records are installed.")
        st.stop()
    chosen_record = st.selectbox("Dataset", available,
                                 index=available.index("ode") if "ode" in available else 0,
                                 format_func=dataset_label)
    a, b = st.columns(2)
    noise = a.number_input("Noise, % of signal standard deviation", min_value=0.0,
                           max_value=10.0, value=0.0, step=0.5)
    seed = int(b.number_input("Random seed", min_value=0, max_value=999, value=0))
    from epde_bench.config import variants
    allowed = [name for name, config in variants().items()
               if config.get("method", "epde") == method] if method != "llm" else []
    if method == "epde":
        st.caption("EPDE presets keep all experimental search settings traceable.")
        variant = st.selectbox("EPDE configuration", allowed)
    elif method == "pysindy":
        variant = st.selectbox("SINDy configuration", allowed, index=0)
        st.caption("Uses the shared EPDE differentiation by default; SINDy's front "
                   "and objective definitions remain distinct from EPDE's.")
    else:
        variant = "llm"
else:
    if source == "Upload data":
        uploaded = st.file_uploader("Numerical data: CSV, NPY or NPZ", type=["csv", "npy", "npz"])
        if uploaded is not None:
            try:
                arrays = parse_upload(uploaded.getvalue(), uploaded.name)
                upload_name = uploaded.name
            except (ValueError, TypeError, OSError, KeyError) as exc:
                st.error(f"Invalid upload: {exc}")
    else:
        st.caption("Enter the RHS of du/dt = f(t, u). The expression is parsed as a restricted "
                   "mathematical syntax and is never executed as Python code.")
        expression = st.text_input("Right-hand side f(t,u)", value="-0.5*u + sin(t)")
        a, b, c = st.columns(3)
        horizon = a.number_input("End time", min_value=1.0, max_value=30.0, value=10.0)
        u0 = b.number_input("Initial value u(0)", min_value=-10.0, max_value=10.0, value=1.0)
        count = c.number_input("Time samples", min_value=60, max_value=3000, value=301, step=20)
        try:
            compile_rhs(expression)
            arrays = generate(expression, horizon, u0, int(count))
            upload_name = "generated_ode"
        except (ValueError, OverflowError) as exc:
            st.error(f"Cannot generate this equation: {exc}")
    if arrays is not None:
        variables = [k for k in arrays if k.startswith("var_")]
        axes = sorted(k for k in arrays if k.startswith("axis_") and k != "axis_names")
        count_values = sum(arrays[k].size for k in variables)
        st.success(f"Loaded {len(variables)} variable(s), {len(axes)} axis/axes, "
                   f"{count_values:,} numerical values")
        if expression is not None:
            st.caption(f"Generated reference ODE: du/dt = {expression}. "
                       "Structural recovery is not scored until the equation is "
                       "translated into EPDE's canonical token representation.")
        with st.expander("Preview data"):
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 2.5))
            for key in variables[:4]:
                y = arrays[key]
                ax.plot(np.asarray(y).ravel()[:2000], label=key[4:])
            ax.set(xlabel="sample", ylabel="value")
            ax.legend()
            show_figure(fig)
        from epde_bench.config import read_yaml
        from epde_bench.paths import CONFIG_DIR
        from epde_bench.poly_options import polynomial_kwargs
        base = read_yaml(CONFIG_DIR / "_base.yaml")
        settings = {"seed": 0, "noise": 0.0, "search": dict(base["search"])}
        effort = st.select_slider("Search effort", options=["Quick", "Standard", "Thorough"],
                                  value="Quick")
        population, epochs = {"Quick": (8, 3), "Standard": (16, 5),
                              "Thorough": (24, 8)}[effort]
        dim = len(axes) - 1
        search = settings["search"]
        search["domain"] = {"boundary_width": "auto"}
        search["evolution"] = dict(search.get("evolution") or {},
                                   population_size=population, training_epochs=epochs)
        search["preprocessing"] = dict(search.get("preprocessing") or {},
                                        max_deriv_order=[2] + [3] * dim)
        search.setdefault("solver", {})["device"] = "cpu"
        search.setdefault("solver", {})["use_solver"] = False
        if method == "epde":
            st.caption("Search budget affects EPDE. PySINDy uses its own STLSQ selection.")
        with st.expander("Differentiation protocol", expanded=False):
            differentiation = st.selectbox("Shared derivative estimator", ["FD", "poly"],
                                           format_func={"FD": "Finite differences",
                                                        "poly": "Local polynomial fit"}.get)
            search["preprocessing"]["default_preprocessor_type"] = differentiation
            search["preprocessing"]["preprocessor_kwargs"] = {}
            if differentiation == "poly":
                a, b = st.columns(2)
                degree = int(a.number_input("Polynomial degree", min_value=3, max_value=8, value=4))
                window = int(b.select_slider("Polynomial window", options=list(range(7, 32, 2)), value=11))
                try:
                    search["preprocessing"]["preprocessor_kwargs"] = polynomial_kwargs(
                        window, degree, [2] + [3] * dim, tuple(len(arrays[k]) for k in axes))
                except ValueError as exc:
                    st.error(str(exc))
                    arrays = None
        if method == "pysindy":
            st.caption("PySINDy will use the exact same differentiation settings as EPDE "
                       "for this upload, but has a different optimizer and objective axes.")
        if method != "llm":
            with st.expander("Effective numerical configuration"):
                st.json(settings)

allow_transfer = True
if method == "llm" and remote_ready:
    allow_transfer = st.checkbox("I agree to send a bounded numerical sample to the configured LLM service")

is_valid = bool(chosen_record) if source == "Built-in record" else arrays is not None
run_allowed = is_valid and (method != "llm" or (remote_ready and allow_transfer))
if st.button("Start paired searches" if paired else "Start discovery",
             type="primary", disabled=not run_allowed):
    try:
        methods = ("epde", "pysindy") if paired else (method,)
        for item in methods:
            queue.check_capacity(owner, item)
        if source == "Built-in record":
            requests = [{"method": item, "source": "record", "dataset": chosen_record,
                         "variant": ({"epde": "default", "pysindy": "pysindy"}[item]
                                     if paired else variant),
                         "noise": noise, "seed": seed} for item in methods]
        else:
            data_path, config_path = queue.save_upload(owner, arrays, settings)
            requests = [{"method": item, "source": "upload", "dataset": "custom",
                         "data_path": str(data_path), "config_path": str(config_path),
                         "seed": 0, "noise": 0.0} for item in methods]
        started = []
        for request in requests:
            jid = queue.submit(owner, request)
            started.append(jid)
        st.session_state["discovery_selected_job"] = started[0]
        st.success(f"Queued {len(started)} search(es). Results are independently recorded.")
    except (ValueError, OSError) as exc:
        st.error(f"Cannot submit the requested search(es): {exc}")

st.divider()
st.subheader("Your jobs")
rows = queue.list_jobs(owner)
if not rows:
    st.info("Start a search to see reproducible results and downloadable records.")
    st.stop()
labels = {j["id"]: f"{j['dataset']} · {j['method']} · {j['state']}" for j in rows}
# Discovery Lab exposes an informative same-input synopsis, never a ranking:
# objectives have different definitions and cannot be placed on common axes.
finished = [j for j in rows if j["state"] == "finished" and queue.result(owner, j["id"])]
if len(finished) >= 2:
    import pandas as pd
    summaries = []
    for j in finished:
        result = queue.result(owner, j["id"])
        if result.get("status") != "ok":
            continue
        payload = queue._read(queue.job_path(owner, j["id"]) / "request.json") or {}
        source_id = (result.get("data_sha256") if j["source"] == "upload"
                     else f"{j['dataset']}|{payload.get('noise')}|{payload.get('seed')}")
        summaries.append({"input": source_id, "method": j["method"],
                          "candidate count": len(result.get("front") or []),
                          "fit seconds": result.get("fit_seconds"),
                          "found equation": " | ".join(result.get("selected") or [])})
    if summaries:
        with st.expander("Completed searches on the same input (descriptive only)"):
            st.dataframe(pd.DataFrame(summaries), hide_index=True, width="stretch")
            st.caption("Rows are not ranked. Fit time is observational; the search budgets "
                       "and objective scales may differ.")
selected = st.session_state.get("discovery_selected_job", rows[0]["id"])
ids = list(labels)
chosen = st.selectbox("Search history", ids, index=ids.index(selected) if selected in ids else 0,
                      format_func=labels.get)
job = next(j for j in rows if j["id"] == chosen)
if job["state"] in ("queued", "running"):
    st.info(f"Status: {job['state']} (server-side CPU job; page refresh is not required for persistence)")
    if st.button("Cancel this search"):
        queue.cancel(owner, chosen)
        st.rerun()
    if st.button("Refresh job status"):
        st.rerun()
    st.stop()
record = queue.result(owner, chosen)
if not record:
    st.warning(f"Job ended with state '{job['state']}' and no result record. "
               "It may have timed out or been stopped.")
    st.stop()
if record.get("status") != "ok":
    st.warning(record.get("reason") or record.get("error") or "Search failed")
else:
    st.markdown("### Discovered equation")
    equations = record.get("selected") or ((record.get("front") or [None])[0])
    from support.equations import system_latex
    axes = (record.get("problem") or {}).get("axis_names") or ["t", "x", "y"]
    if equations:
        try:
            st.latex(system_latex(equations, axes))
        except (ValueError, IndexError, KeyError):
            st.code("\n".join(equations), language=None)

    st.markdown("### Candidate front")
    objective_labels = record.get("criterion_labels") or ["objective 1", "objective 2"]
    st.caption(f"Axes: {objective_labels[0]} / {objective_labels[1]}. "
               "Objectives are method-specific; do not compare their values across methods.")
    front, objectives = record.get("front") or [], record.get("objectives") or []
    points = [(i, o) for i, o in enumerate(objectives) if o is not None and len(o) >= 2]
    if points:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.scatter([o[0] for _, o in points], [o[1] for _, o in points])
        for i, o in points:
            ax.annotate(str(i), (o[0], o[1]))
        ax.set_xlabel(objective_labels[0]); ax.set_ylabel(objective_labels[1]); ax.grid(alpha=.25)
        show_figure(fig)
        if len(points) == 1:
            st.caption("Only one candidate was emitted, so this is a point, not a Pareto frontier.")
    else:
        st.info("This method did not report two-dimensional objective values.")

    st.markdown("### Data vs reconstructed equation")
    try:
        from support.viewer import record_problem, reconstruction_inputs
        from epde_bench.reconstruction import plot_reconstruction
        problem = record_problem(record)
        data, search = reconstruction_inputs(record, problem)
        system = equations
        if not system:
            raise ValueError("No selected equation")
        fig, scores = plot_reconstruction(problem, system, search, data=data)
        show_figure(fig)
        st.caption("This is an in-sample reconstruction on the available record, "
                   "not evidence of forecasting/generalization to unseen data.")
    except Exception as exc:
        st.info(f"Reconstruction is unavailable for this equation form: {type(exc).__name__}")
    if len(front) > 1:
        with st.expander("All equation candidates"):
            for i, candidate in enumerate(front):
                st.code(f"[{i}]\n" + "\n".join(candidate), language=None)

# Export a portable scientific record without leaking the private owner token
# embedded in the server's absolute upload path.
export_record = dict(record)
if export_record.get("dataset") == "custom":
    export_record["data_file"] = "input_data.npz"
    st.caption("For a portable custom-data result, export both JSON and NPZ. "
               "The recorded SHA-256 identifies the original input.")
    try:
        from pathlib import Path
        path = Path(record["data_file"])
        if path.is_file() and path.is_relative_to(queue.owner_path(owner).resolve()):
            st.download_button("Download input_data.npz", data=path.read_bytes(),
                               file_name="input_data.npz", mime="application/octet-stream")
    except (KeyError, OSError, ValueError):
        pass
st.download_button("Download reproducible result JSON",
                   data=json.dumps(export_record, indent=2, ensure_ascii=False),
                   file_name=f"discovery_{chosen}.json", mime="application/json")
with st.expander("Result metadata"):
    st.json({"method": record.get("method"), "schema_version": record.get("schema_version"),
             "status": record.get("status"), "fit_seconds": record.get("fit_seconds"),
             "wall_seconds": record.get("wall_seconds"),
             "criterion_labels": record.get("criterion_labels"),
             "configuration": record.get("search_config") or record.get("config")})
