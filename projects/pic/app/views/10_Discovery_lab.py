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

# ---------------------------------------------------------------- session ---
# The session code is kept in the address bar, so a reload or a bookmark
# reopens the same experiments. Anyone with the link sees them: share it only
# when you want to.
code = st.query_params.get("session", "")
try:
    queue.owner_path(code)
except ValueError:
    code = st.session_state.get("discovery_lab_owner") or queue.new_owner()
    st.query_params["session"] = code
st.session_state["discovery_lab_owner"] = owner = code

with st.expander("Your session"):
    st.caption("Your experiments are stored on the server under this code, which is also in the "
               "page address: bookmark the page to come back. Anyone with the address sees the "
               "results, so share it only on purpose.")
    st.code(owner, language=None)
    other = st.text_input("Open another session by its code", type="password")
    if st.button("Open session") and other:
        try:
            if not queue.owner_path(other.strip()).exists():
                raise ValueError
            st.query_params["session"] = other.strip()
            st.session_state["discovery_lab_owner"] = other.strip()
            st.rerun()
        except ValueError:
            st.error("No session with this code.")

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


tab_new, tab_results, tab_compare = st.tabs(["① New experiment", "② Results", "③ Compare"])

# ------------------------------------------------------------ new experiment
with tab_new:
    st.markdown("#### 1 · Data")
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
        problem = load_problem(record_name)
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

    st.markdown("#### 2 · Methods")
    options = ["epde", "pysindy"] + (["llm"] if LLM_READY else [])
    methods = st.multiselect("Run on the same input", options, default=["epde", "pysindy"],
                             format_func=METHOD_LABELS.get)
    if not LLM_READY:
        st.caption("The LLM agent slot is ready but no service is configured on this server.")
    elif "llm" in methods:
        st.warning("The LLM service receives a sample of these data. Do not send private measurements.")

    st.markdown("#### 3 · Settings")
    effort = st.select_slider("EPDE search budget", list(EFFORTS), value="standard",
                              format_func=EFFORTS.get,
                              help="Larger budgets find more but take longer. PySINDy has no budget.")
    settings = None
    if arrays is not None:
        with st.expander("Search space and derivatives", expanded=source != "Built-in equation"):
            if source == "Generate an ODE":
                st.caption("Set from your expression; change them to see how the search space matters.")
            c1, c2, c3 = st.columns(3)
            derivatives = c1.selectbox("Derivatives from data", ["FD", "poly"],
                                       format_func={"FD": "finite differences",
                                                    "poly": "local polynomial fit"}.get,
                                       help="The same derivatives are given to every method.")
            time_order = c2.selectbox("Highest time derivative", [1, 2],
                                      index=[1, 2].index(space.get("time_order", 2)))
            power = c3.selectbox("Highest power of a variable", [1, 2, 3],
                                 index=[1, 2, 3].index(space.get("power", 3)))
            c4, c5 = st.columns(2)
            space_order = None
            if space.get("space_order"):
                space_order = c4.selectbox("Highest space derivative", [1, 2, 3, 4], index=2)
            time_terms = c4.checkbox("Include t and t² as terms", value=space.get("time_terms", True),
                                     help="Lets the equation depend explicitly on time (and on the "
                                          "space coordinate for PDEs). Switch off for autonomous laws: "
                                          "otherwise a method may imitate a missing term with t.")
            use_trig = c5.checkbox("Include sin and cos of t", value=space.get("frequency") is not None)
            frequency = (c5.number_input("Frequency of sin/cos", 0.01, 50.0,
                                         float(space.get("frequency") or 1.0))
                         if use_trig else None)
        from epde_bench.config import read_yaml
        from epde_bench.paths import CONFIG_DIR
        base = read_yaml(CONFIG_DIR / "_base.yaml")
        search = dict(base["search"])
        search["domain"] = {"boundary_width": "auto"}
        population, epochs = {"quick": (8, 3), "standard": (16, 5), "thorough": (24, 8)}[effort]
        search["evolution"] = dict(search.get("evolution") or {}, population_size=population,
                                   training_epochs=epochs)
        orders = [time_order] + ([space_order] if space_order else [])
        search["preprocessing"] = dict(search.get("preprocessing") or {}, max_deriv_order=orders,
                                       default_preprocessor_type=derivatives, preprocessor_kwargs={})
        if derivatives == "poly":
            from epde_bench.poly_options import polynomial_kwargs
            try:
                shape = tuple(len(arrays[f"axis_{i}"]) for i in range(len(orders)))
                search["preprocessing"]["preprocessor_kwargs"] = polynomial_kwargs(11, 4, orders, shape)
            except ValueError as exc:
                st.error(str(exc))
                arrays = None
        tokens = [{"family": "grid", "labels": ["x"], "max_power": 2}] if time_terms else []
        if frequency:
            # meaningful: sin/cos may stand alone as a forcing term, as for the Duffing record
            tokens.append({"family": "fixed_trigonometric", "freq": float(frequency),
                           "meaningful": True})
        search["search_space"] = dict(search.get("search_space") or {}, data_fun_pow=power,
                                      tokens=tokens)
        search.setdefault("solver", {})["use_solver"] = False
        settings = {"seed": 0, "noise": float(noise), "search": search, "effort": effort,
                    "label": label, "origin": origin}
        if reference:
            settings["reference_truth"] = reference

    ready = (record_name is not None or arrays is not None) and bool(methods)
    if st.button(f"Run {len(methods)} search(es) on this input" if methods else "Choose a method",
                 type="primary", disabled=not ready):
        try:
            for method in methods:
                queue.check_capacity(owner, method, count=len(methods))
            if record_name:
                key = f"record|{record_name}|{noise:g}|{seed}"
                requests = [{"method": m, "source": "record", "dataset": record_name,
                             "variant": "pysindy" if m == "pysindy" else "default",
                             "noise": float(noise), "seed": seed, "effort": effort,
                             "display": {"label": label, "input_key": key}} for m in methods]
            else:
                key = f"upload|{arrays_digest(arrays)}|{noise:g}"
                data_path, config_path = queue.save_upload(owner, arrays, settings)
                requests = [{"method": m, "source": "upload", "dataset": "custom",
                             "data_path": str(data_path), "config_path": str(config_path),
                             "noise": float(noise), "seed": 0, "effort": effort,
                             "display": {"label": label, "input_key": key}} for m in methods]
            ids = [queue.submit(owner, request) for request in requests]
            st.session_state["discovery_selected_job"] = ids[0]
            st.session_state["discovery_compare_key"] = key
            st.success(f"Started {len(ids)} search(es). Follow them in **② Results**; when they "
                       "finish, **③ Compare** shows them side by side.")
        except (ValueError, OSError) as exc:
            st.error(f"Cannot start: {exc}")

# ------------------------------------------------------------------ results
STATE_TEXT = {"queued": "⏳ waiting", "running": "⚙️ running", "finished": "✅ done",
              "error": "❌ failed", "timeout": "⌛ time limit", "stopped": "⏹ stopped",
              "interrupted": "⚠️ interrupted"}


def job_table(rows):
    return pd.DataFrame([{
        "data": r.get("label") or r.get("dataset"),
        "method": ui.METHOD_NAMES.get(r.get("method"), r.get("method")),
        "noise, %": r.get("noise", 0), "seed": r.get("seed", 0),
        "budget": r.get("effort", "standard"),
        "status": STATE_TEXT.get(r.get("state"), r.get("state")),
        "time, s": (round((r.get("ended") or time.time()) - r["started"])
                    if r.get("started") else None),
    } for r in rows])


@st.fragment(run_every=4)
def live_status():
    rows = queue.list_jobs(owner)
    active = [r for r in rows if r.get("state") in ("queued", "running")]
    before = st.session_state.get("discovery_active", set())
    now = {r["id"] for r in active}
    st.session_state["discovery_active"] = now
    if before - now:
        st.rerun(scope="app")                     # something finished: refresh the whole page
    if active:
        st.dataframe(job_table(active), hide_index=True, width="stretch")
        st.caption("Searches run on the server; you can leave or reload the page.")


with tab_results:
    rows = queue.list_jobs(owner)
    if not rows:
        st.info("No experiments yet. Start one in **① New experiment**.")
    else:
        live_status()
        st.dataframe(job_table(rows), hide_index=True, width="stretch")
        ids = [r["id"] for r in rows]
        names = {r["id"]: f"{r.get('label') or r.get('dataset')} · "
                          f"{ui.METHOD_NAMES.get(r.get('method'), r.get('method'))} · "
                          f"noise {r.get('noise', 0):g} % · {STATE_TEXT.get(r.get('state'), '')}"
                 for r in rows}
        current = st.session_state.get("discovery_selected_job")
        chosen = st.selectbox("Open a result", ids, index=ids.index(current) if current in ids else 0,
                              format_func=names.get)
        st.session_state["discovery_selected_job"] = chosen
        job = next(r for r in rows if r["id"] == chosen)
        record = queue.result(owner, chosen)
        if job.get("state") in ("queued", "running"):
            st.info("This search is still running.")
            if st.button("Cancel this search"):
                queue.cancel(owner, chosen)
                st.rerun()
        elif not record:
            st.warning(f"No result: {STATE_TEXT.get(job.get('state'), job.get('state'))}. "
                       "The search may have hit the time limit; try a quicker budget or fewer methods.")
        elif record.get("status") != "ok":
            st.warning(record.get("reason") or record.get("error") or "The search failed.")
        else:
            axes = ui.axes_of(record)
            truth = (record.get("problem") or {}).get("truth")
            found = record.get("selected") or []
            if truth:
                left, right = st.columns(2)
                with left:
                    st.markdown("**Known law**")
                    ui.latex(truth, axes)
                with right:
                    st.markdown(f"**Found by {ui.METHOD_NAMES.get(record.get('method'))}**")
                    ui.latex(found, axes)
                kind, text = ui.verdict(record)
                getattr(st, kind)(text)
                ui.term_table(record)
            else:
                st.markdown(f"**Found by {ui.METHOD_NAMES.get(record.get('method'))}**")
                ui.latex(found, axes)
                st.caption("No known law for these data: judge the equation by how its solution "
                           "matches the data below.")
            scores = ui.solution_scores(record)
            cols = st.columns(max(1, len(scores)) + 3)
            for col, (var, (r2, nrmse)) in zip(cols, scores.items()):
                col.metric(f"solution R², {var}", ui._fmt(r2),
                           help="The equation is solved from the first data point and compared "
                                "with the data (1 = perfect).")
            offset = len(scores) if scores else 1
            cols[offset].metric("equation balance R²", ui._fmt(ui.balance_score(record)),
                                help="Left-hand side computed from the data against the right-hand "
                                     "side, with the derivatives used by the search.")
            cols[offset + 1].metric("candidates", len(record.get("front") or []))
            cols[offset + 2].metric("search time", ui._fmt(record.get("fit_seconds"), "{:.1f} s"))
            st.markdown("**Solution of the equation against the data**")
            ui.solution_plots([(ui.METHOD_NAMES.get(record.get("method")), record)])
            st.caption("In-sample check on the same record the method saw; it is not a forecast.")
            with st.expander("Candidates (Pareto front)"):
                ui.pareto(record)
                for i, candidate in enumerate(record.get("front") or []):
                    st.markdown(f"`[{i}]`" + (" — selected" if candidate == found else ""))
                    ui.latex(candidate, axes)
        if record:
            export = {k: v for k, v in record.items() if k != "data_file"}
            st.download_button("Download result (JSON)", json.dumps(export, indent=1, ensure_ascii=False),
                               f"discovery_{chosen[:8]}.json", "application/json")
        if job.get("state") not in ("queued", "running"):
            if st.button("Delete this result"):
                queue.delete_job(owner, chosen)
                st.session_state.pop("discovery_selected_job", None)
                st.rerun()

# ------------------------------------------------------------------ compare
with tab_compare:
    rows = [r for r in queue.list_jobs(owner) if r.get("state") == "finished"]
    results = {r["id"]: queue.result(owner, r["id"]) for r in rows}
    rows = [r for r in rows if (results.get(r["id"]) or {}).get("status") == "ok"]
    if not rows:
        st.info("Finished searches appear here, grouped by input. Run several methods on the same "
                "data in **① New experiment** to compare them.")
    else:
        groups = {}
        for r in rows:
            groups.setdefault(r.get("input_key") or r["id"], []).append(r)
        keys = list(groups)
        preferred = st.session_state.get("discovery_compare_key")
        key = st.selectbox(
            "Input", keys, index=keys.index(preferred) if preferred in keys else 0,
            format_func=lambda k: (f"{groups[k][0].get('label') or groups[k][0].get('dataset')} · "
                                   f"noise {groups[k][0].get('noise', 0):g} % · {len(groups[k])} run(s)"))
        members = groups[key]
        names = {r["id"]: f"{ui.METHOD_NAMES.get(r.get('method'))} · {r.get('effort', 'standard')} budget"
                          f" · started {time.strftime('%H:%M:%S', time.localtime(r.get('created', 0)))}"
                 for r in members}
        picked = st.multiselect("Runs", list(names), default=list(names), format_func=names.get)
        chosen = [(names[i], results[i]) for i in picked]
        if chosen:
            first = chosen[0][1]
            truth = (first.get("problem") or {}).get("truth")
            axes = ui.axes_of(first)
            if truth:
                st.markdown("**Known law**")
                ui.latex(truth, axes)
            cards = st.columns(len(chosen))
            for card, (title, record) in zip(cards, chosen):
                with card:
                    st.markdown(f"**{title}**")
                    ui.latex(record.get("selected") or [], axes)
                    judged = ui.verdict(record)
                    if judged:
                        getattr(st, judged[0])(judged[1])
            st.markdown("**Summary** (descriptive, in the order of the runs; not a ranking)")
            table = pd.DataFrame([dict(run=title, **ui.summary_row({}, record))
                                  for title, record in chosen])
            st.dataframe(table, hide_index=True, width="stretch",
                         column_config={"coefficient error": st.column_config.NumberColumn(format="percent"),
                                         "solution R² (worst variable)": st.column_config.NumberColumn(format="%.4f"),
                                         "equation balance R²": st.column_config.NumberColumn(format="%.4f"),
                                         "search time, s": st.column_config.NumberColumn(format="%.1f")})
            st.caption("Each method selects its equation by its own criteria, so their internal "
                       "scores are not comparable. The solution and balance R² are computed the same "
                       "way for every method from the same data. One input and one seed say little "
                       "about a method in general.")
            st.markdown("**Solutions against the data**")
            ui.solution_plots(chosen)
