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
    names = {r["id"]: f"{ui.job_method_label(r)} · {r.get('effort', 'standard')} budget"
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
        equation_tab,metrics_tab,plots_tab,balance_tab=st.tabs(
            ["Equations","Metrics","Solution fit","Equation balance"])
        with equation_tab:
            truth=(first.get("problem") or {}).get("truth")
            if truth:
                st.markdown("**Known reference law**")
                ui.latex(truth,axes)
            cards=st.columns(len(chosen),gap="medium")
            for card,(name,record) in zip(cards,chosen):
                with card:
                    with st.container(border=True):
                        st.markdown("**"+ui.method_label(record)+"**")
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
                         column_config=ui.metric_columns())
            st.caption("Only matching input fingerprints are grouped. HD and RE follow the "
                       "PDEComp benchmark and are computed here in the same way for every method, "
                       "against the known law of this record. The methods' objectives and derivative "
                       "estimators differ; internal scores are not interchangeable. Solution R² is "
                       "in-sample, not a forecast.")
            for method in sorted(methods_in_view, key=str):
                st.caption("**"+ui.METHOD_NAMES.get(method, method)+"**: "+ui.PROTOCOL_NOTES.get(method, ""))
        with plots_tab:
            st.caption("Plots show solution reconstruction against the original data.")
            ui.solution_plots(chosen)
        with balance_tab:
            st.caption("Target computed from the data against each equation's right-hand side.")
            ui.balance_plots(chosen)
        st.caption("Comparison groups require matching source/data fingerprints, "
                   "loader settings and noise realization; identical titles or seeds "
                   "alone are insufficient. Native imports are consistency-checked but "
                   "not independently rerun. Multiple seeds are needed for strong claims.")

# ------------------------------------------------------------------ series
all_rows = [r for r in queue.list_jobs(owner)
            if r.get("state") in ("finished", "error", "timeout")]
by_dataset = {}
for r in all_rows:
    by_dataset.setdefault(r.get("dataset"), []).append(r)
repeated = {d: rs for d, rs in by_dataset.items()
            if d != "custom" and len({(r.get("seed"), r.get("noise")) for r in rs}) > 1}
st.divider()
st.subheader("Series over seeds and noise levels")
if not repeated:
    st.caption("Run a record with several seeds or noise levels (Run a search, step 1) to see "
               "success rates with 95 % intervals and the noise limit of each method.")
else:
    from support.discovery_lab import series
    dataset = st.selectbox("Record", list(repeated),
                           format_func=lambda d: repeated[d][0].get("label") or d,
                           key="series_dataset")
    pairs = [(r, queue.result(owner, r["id"])) for r in repeated[dataset]]
    table, limits = series.summarize(pairs, ui.job_method_label)
    if not table:
        st.info("This record has no known law, so success cannot be counted.")
    else:
        frame = pd.DataFrame(table)
        st.dataframe(frame, hide_index=True, width="stretch", column_config={
            "success rate": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
            "HD, mean": st.column_config.NumberColumn(format="%.2f", help=ui.METRIC_HELP["HD"]),
            "RE (sum), mean of successes": st.column_config.NumberColumn(
                format="%.4g", help=ui.METRIC_HELP["RE (sum)"]),
            "median time, s": st.column_config.NumberColumn(format="%.2f")})
        cols = st.columns(max(1, len(limits)))
        for col, (method, limit) in zip(cols, limits.items()):
            col.metric(f"noise limit · {method}", "below the lowest level" if limit is None
                       else f"{limit:g} %",
                       help="Highest tested noise level up to which at least 3 of 5 runs (60 %) "
                            "found the law at every level, as in PDEComp. Needs at least 3 runs "
                            "per level.")
        if frame["noise, %"].nunique() > 1:
            import matplotlib.pyplot as plt
            from support.common import show_figure
            fig, ax = plt.subplots(figsize=(7, 3))
            for method, part in frame.groupby("method"):
                ax.plot(part["noise, %"], part["success rate"], marker="o", label=method)
            ax.axhline(series.LIMIT_RATE, color="grey", lw=.8, ls="--")
            ax.set_xlabel("noise, % of the standard deviation")
            ax.set_ylabel("success rate")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(alpha=.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
            show_figure(fig)
        st.caption("A run succeeds when its selected equation has exactly the terms of the known "
                   "law. Failed runs (errors, time limit) count as unsuccessful, as in PDEComp. "
                   "Methods are described side by side, not ranked.")

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
