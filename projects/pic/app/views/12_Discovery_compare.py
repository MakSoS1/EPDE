"""Side-by-side method results on one identical numerical input.

No synthetic leaderboard: each optimizer has its own objective definition.
"""
from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from support.discovery_lab import job_queue as queue, ui
from support.discovery_lab.session import current_owner

from support.minimal_ui import intro
intro("Compare", "Compare recovered equations on the same verified numerical input.")
owner = current_owner()
st.page_link("views/10_Discovery_lab.py", label="Start another search", icon=":material/search:")

rows = [r for r in queue.list_jobs(owner) if r.get("state") == "finished"]
results = {r["id"]: queue.result(owner, r["id"]) for r in rows}
rows = [r for r in rows if (results.get(r["id"]) or {}).get("status") == "ok"]
if not rows:
    st.info("Finished searches appear here, grouped by input. Run several methods on the same "
            "data in **Run a search** to compare them.")
else:
    # Never equate inputs merely by their dataset title, noise and seed.
    # A matching group requires original data/content hashes, scientific
    # source identity, loader arguments and noise-realisation protocol.
    from support.discovery_lab.portable import input_key
    groups = {}
    for r in rows:
        verified = input_key(results[r["id"]])
        groups.setdefault(verified or ("unverifiable:" + r["id"]), []).append(r)
    keys = list(groups)
    preferred = st.session_state.get("discovery_compare_key")
    input_label = lambda k: (f"{groups[k][0].get('label') or groups[k][0].get('dataset')} · "
                             f"noise {groups[k][0].get('noise', 0):g} % · {len(groups[k])} run(s)")
    key = st.selectbox("Input group", keys,
                       index=keys.index(preferred) if preferred in keys else 0,
                       format_func=input_label)
    members = groups[key]
    if key.startswith("unverifiable:"):
        st.warning("No reliable input fingerprint: this run is isolated from "
                   "all other experiments rather than assuming identical data.")
    if any((results[r["id"]] or {}).get("execution_location") == "user_computer"
           for r in members):
        st.caption("Imported results were generated on users' computers. "
                   "Their file/source hashes are self-reported and verified for "
                   "internal consistency; this website has not independently "
                   "recomputed the underlying heavy-data experiment.")
    names = {r["id"]: f"{ui.METHOD_NAMES.get(r.get('method'))} · {r.get('effort', 'standard')} budget"
                      f" · started {time.strftime('%H:%M:%S', time.localtime(r.get('created', 0)))}"
             for r in members}
    with st.container(border=True):
        a,b,c=st.columns(3)
        a.metric("Dataset",members[0].get("label") or members[0].get("dataset"))
        b.metric("Noise",f"{float(members[0].get('noise',0)):g}%")
        c.metric("Experiments",len(members))
        st.caption("Input fingerprint: "+key[:18]+"…" if not key.startswith("unverifiable:")
                   else "Numerical input fingerprint unavailable.")
    picked=st.multiselect("Select runs",list(names),
                         default=list(names)[:2],max_selections=3,format_func=names.get)
    chosen=[(names[i],results[i]) for i in picked]
    if chosen:
        first=chosen[0][1]
        axes=ui.axes_of(first)
        equation_tab,metrics_tab,plots_tab=st.tabs(
            ["Equations","Metrics","Solution fit"])
        with equation_tab:
            truth=(first.get("problem") or {}).get("truth")
            if truth:
                st.markdown("**Known reference law**")
                ui.latex(truth,axes)
            cards=st.columns(len(chosen),gap="medium")
            for card,(name,record) in zip(cards,chosen):
                with card:
                    with st.container(border=True):
                        method=ui.METHOD_NAMES.get(record.get("method"),record.get("method"))
                        st.markdown("**"+str(method)+"**")
                        st.caption(name)
                        ui.latex(record.get("selected") or [],axes)
                        verdict=ui.verdict(record)
                        if verdict:
                            getattr(st,verdict[0])(verdict[1])
                        st.caption(f"Search time: {record.get('fit_seconds',0):.1f} s"
                                   if isinstance(record.get("fit_seconds"),(int,float))
                                   else "Search time unavailable")
        with metrics_tab:
            methods_in_view={rec.get("method") for _,rec in chosen}
            summaries=[dict(run=name,**ui.summary_row({},record))
                       for name,record in chosen]
            if len(methods_in_view)>1:
                for row in summaries:
                    row["equation balance R²"]=None
            st.dataframe(pd.DataFrame(summaries),hide_index=True,width="stretch",
                         column_config={
                             "coefficient error":st.column_config.NumberColumn(format="percent"),
                             "solution R² (worst variable)":st.column_config.NumberColumn(format="%.4f"),
                             "equation balance R²":st.column_config.NumberColumn(format="%.4f"),
                             "search time, s":st.column_config.NumberColumn(format="%.1f")})
            st.caption("Only matching input fingerprints are grouped. EPDE and PySINDy "
                       "objectives and derivative estimators differ; internal scores "
                       "are not interchangeable. Solution R² is in-sample, not a forecast.")
        with plots_tab:
            st.caption("Plots show solution reconstruction against the original data.")
            ui.solution_plots(chosen)
        st.caption("Comparison groups require matching source/data fingerprints, "
                   "loader settings and noise realization; identical titles or seeds "
                   "alone are insufficient. Native imports are consistency-checked but "
                   "not independently rerun. Multiple seeds are needed for strong claims.")

# Curated published benchmark archives remain available without exposing
# other visitors' private files from the shared app/results directory.
if st.checkbox("Show published benchmark archives", value=False,
               key="compare_show_archives"):
    st.subheader("Published benchmark archive (read-only)")
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
