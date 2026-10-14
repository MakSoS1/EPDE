"""Saved Lab experiments: progress, equation, candidates, plots and downloads."""
from __future__ import annotations

import json
import time

import pandas as pd
import streamlit as st

from support.discovery_lab import job_queue as queue
from support.discovery_lab import ui
from support.discovery_lab.session import current_owner

st.title("🗂️ Results")
owner = current_owner()
st.caption("All your saved Lab searches appear here, including their data, methods, "
           "status and scientific results. Open a result to inspect or download it.")
st.page_link("views/10_Discovery_lab.py", label="New experiment in Lab", icon="🧪")

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

