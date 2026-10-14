"""Side-by-side method results on one identical numerical input.

No synthetic leaderboard: each optimizer has its own objective definition.
"""
from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from support.discovery_lab import job_queue as queue, ui
from support.discovery_lab.session import current_owner

st.title("⚖️ Compare")
owner = current_owner()
st.caption("Choose experiments on the same input to inspect recovered equations, "
           "prediction quality and runtimes side by side. This is not a framework ranking.")
st.page_link("views/10_Discovery_lab.py", label="Run a search", icon="🚀")
st.caption("When you choose both methods in Run a search, they share a dataset, 
           "noise level and seed. Only matching input groups can be compared.")

rows = [r for r in queue.list_jobs(owner) if r.get("state") == "finished"]
results = {r["id"]: queue.result(owner, r["id"]) for r in rows}
rows = [r for r in rows if (results.get(r["id"]) or {}).get("status") == "ok"]
if not rows:
    st.info("Finished searches appear here, grouped by input. Run several methods on the same "
            "data in **Run a search** to compare them.")
else:
    groups = {}
    for r in rows:
        groups.setdefault(r.get("input_key") or r["id"], []).append(r)
    keys = list(groups)
    preferred = st.session_state.get("discovery_compare_key")
    input_label = lambda k: (f"{groups[k][0].get('label') or groups[k][0].get('dataset')} · "
                             f"noise {groups[k][0].get('noise', 0):g} % · {len(groups[k])} run(s)")
    if len(keys) <= 6:
        key = st.pills("Input", keys, default=preferred if preferred in keys else keys[0],
                       selection_mode="single", format_func=input_label) or keys[0]
    else:
        key = st.selectbox("Input", keys,
                           index=keys.index(preferred) if preferred in keys else 0,
                           format_func=input_label)
    members = groups[key]
    names = {r["id"]: f"{ui.METHOD_NAMES.get(r.get('method'))} · {r.get('effort', 'standard')} budget"
                      f" · started {time.strftime('%H:%M:%S', time.localtime(r.get('created', 0)))}"
             for r in members}
    if len(names) <= 6:
        picked = st.pills("Runs", list(names), default=list(names)[:2],
                          selection_mode="multi", format_func=names.get)
        picked = list(picked or [])[:4]
    else:
        picked = st.multiselect("Runs", list(names), default=list(names)[:2],
                                max_selections=4, format_func=names.get)
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

# Curated published benchmark archives remain available without exposing
# other visitors' private files from the shared app/results directory.
with st.expander("Published benchmark archive (read-only)", expanded=False):
    from pathlib import Path
    from support.common import PIC_DIR

    published_root = PIC_DIR / "experiments"
    folders = sorted(
        (p for p in published_root.iterdir() if p.is_dir() and (p / "campaign.json").is_file()),
        key=lambda p: p.name,
    ) if published_root.exists() else []
    if not folders:
        st.caption("No published benchmark campaigns are bundled with this build.")
    else:
        selected_archive = st.selectbox("Published campaign", folders, format_func=lambda p: p.name)
        report = selected_archive / "report"
        summary_file = report / "success_front.csv"
        if summary_file.is_file():
            archived = pd.read_csv(summary_file)
            st.dataframe(archived, hide_index=True, width="stretch")
            st.caption("Archival campaign metrics were computed by the original "
                       "benchmark protocol. They are not mixed with current Lab run objectives.")
        elif (report / "summary.md").is_file():
            st.markdown((report / "summary.md").read_text(encoding="utf-8"))
        else:
            st.info("This archived campaign has no generated public report.")
