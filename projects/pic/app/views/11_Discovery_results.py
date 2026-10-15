"""Saved Lab experiments: progress, equation, candidates, plots and downloads."""
from __future__ import annotations

import json
import time
import zipfile

import pandas as pd
import streamlit as st

from support.discovery_lab import job_queue as queue
from support.discovery_lab import ui
from support.discovery_lab.session import current_owner

from support.minimal_ui import intro
intro("Results", "Review experiments, open discovered equations, or import a local run.")
owner = current_owner()
st.page_link("views/10_Discovery_lab.py", label="Start a new search", icon=":material/search:")

with st.expander("Import completed experiments · experiment-results.zip", expanded=False):
    st.caption("Upload experiment-results.zip from the local launcher. File metadata "
               "is checked for internal consistency; scientific results are not rerun.")
    archive = st.file_uploader("Import experiment-results.zip",
                               type=["zip"], key="import_portable_results",
                               help="One or two native EPDE/PySINDy results (max 2 MB).")
    if archive is not None and st.button("Import these results", type="primary"):
        try:
            from support.discovery_lab.portable import read_results, input_key
            records = read_results(archive.getvalue())
            if len(queue.list_jobs(owner)) + len(records) > queue.MAX_OWNER_JOBS:
                raise ValueError("Too many saved experiments in this session")
            jobs = []
            for record in records:
                dataset = record["dataset"]
                import_request = {
                    "method": record["method"],
                    "source": "record",
                    "dataset": dataset,
                    "variant": record.get("variant", "default"),
                    "noise": record["noise"], "seed": record["seed"],
                    "effort": record.get("effort", "standard"),
                    "display": {
                        "label": (record.get("problem") or {}).get("title", dataset),
                        "input_key": input_key(record),
                    },
                }
                jobs.append(queue.save_client_result(owner, import_request, record))
            st.success(f"Imported {len(jobs)} scientifically identified result(s). "
                       "They are now available below and under Compare.")
        except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
            st.error("Result import rejected: " + str(exc))
st.divider()

# ------------------------------------------------------------------ results
STATE_TEXT = {"queued": "⏳ waiting", "running": "⚙️ running", "finished": "✅ done",
              "error": "❌ failed", "timeout": "⌛ time limit", "stopped": "⏹ stopped",
              "interrupted": "⚠️ interrupted"}


def job_table(rows):
    return pd.DataFrame([{
        "Experiment": r.get("label") or r.get("dataset"),
        "Method": ui.METHOD_NAMES.get(r.get("method"), r.get("method")),
        "Status": STATE_TEXT.get(r.get("state"), r.get("state")),
        "Time (s)": (round((r.get("ended") or time.time()) - r["started"])
                     if r.get("started") else None),
        "Seed": r.get("seed", 0),
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
        st.caption(f"{len(active)} experiment(s) running or queued; this list updates automatically.")


rows = queue.list_jobs(owner)
if not rows:
    st.info("No experiments yet. Start in Run a search, or import a local results ZIP.")
else:
    live_status()
    a,b,c=st.columns(3)
    a.metric("Experiments",len(rows))
    b.metric("Completed",sum(r.get("state")=="finished" for r in rows))
    c.metric("In progress",sum(r.get("state") in ("queued","running") for r in rows))
    search = st.text_input("Find an experiment", placeholder="Dataset or method", key="results_search")
    status_filter=st.pills("Status",["All","Completed","Running","Failed"],default="All",
                           selection_mode="single",key="results_status") or "All"
    filtered=[r for r in rows if (
        not search or search.casefold() in str(
            (r.get("label") or r.get("dataset") or "")+" "+
            ui.METHOD_NAMES.get(r.get("method"),r.get("method",""))).casefold())
        and (status_filter=="All"
             or (status_filter=="Completed" and r.get("state")=="finished")
             or (status_filter=="Running" and r.get("state") in ("queued","running"))
             or (status_filter=="Failed" and r.get("state") in ("error","timeout","interrupted","stopped")))]
    if not filtered:
        st.caption("No matching experiments.")
        filtered=rows
    st.dataframe(job_table(filtered),hide_index=True,width="stretch")
    ids=[r["id"] for r in filtered]
    names={r["id"]:f"{r.get('label') or r.get('dataset')} · "+
           f"{ui.METHOD_NAMES.get(r.get('method'),r.get('method'))} · "+
           f"{STATE_TEXT.get(r.get('state'),r.get('state'))}" for r in filtered}
    current=st.session_state.get("discovery_selected_job")
    chosen=st.selectbox("Inspect result",ids,
                        index=ids.index(current) if current in ids else 0,
                        format_func=names.get,key="results_selected")
    st.session_state["discovery_selected_job"]=chosen
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
        # Scientific options are recorded per method. Never show EPDE
        # population/epochs as if they controlled PySINDy's optimizer.
        cfg = record.get("config") or {}
        search = record.get("search_config") or cfg.get("search") or {}
        pool = search.get("search_space") or {}
        if record.get("method") == "pysindy":
            opts = cfg.get("pysindy") or {}
            st.caption(
                f"PySINDy STLSQ · threshold: {opts.get('threshold', 'auto')} · "
                f"derivatives: {opts.get('differentiation', 'epde')} · "
                f"max non-zero terms: {pool.get('equation_terms_max_number', 10)} · "
                f"holdout: {float(opts.get('validation_fraction', 0.20)):.0%}")
            with st.expander("PySINDy configuration actually used"):
                st.json({"pysindy": opts, "search_space": pool,
                         "preprocessing": search.get("preprocessing")})
        elif record.get("method") == "epde":
            evo = search.get("evolution") or {}
            st.caption(
                f"EPDE evolution · population: {evo.get('population_size', 'default')} · "
                f"epochs: {evo.get('training_epochs', 'default')} · "
                f"maximum terms: {pool.get('equation_terms_max_number', 10)}")
            with st.expander("EPDE configuration actually used"):
                st.json(search)

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
        with st.expander("Solution fit and diagnostics", expanded=False):
            st.caption("In-sample reconstruction on the search data, not independent forecasting.")
            ui.solution_plots([(ui.METHOD_NAMES.get(record.get("method")), record)])
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
        if st.button("Delete this result", type="secondary"):
            queue.delete_job(owner, chosen)
            st.session_state.pop("discovery_selected_job", None)
            st.rerun()

