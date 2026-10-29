"""Side-by-side method results on one identical numerical input.

No synthetic leaderboard: each optimizer has its own objective definition.
"""
from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from support.discovery_lab import job_queue as queue, ui
from support.discovery_lab.comparison_insights import analyze_comparison
from support.discovery_lab.session import current_owner

from support.minimal_ui import intro
intro("Compare", "Compare recovered equations on the same verified numerical input.")
owner = current_owner()
st.page_link("views/10_Discovery_lab.py", label="Start another search", icon=":material/search:")

rows = [r for r in queue.list_jobs(owner)
        if r.get("state") in ("finished", "error", "timeout", "interrupted")]
results = {r["id"]: queue.result(owner, r["id"]) for r in rows}
rows = [r for r in rows if isinstance(results.get(r["id"]), dict) and results[r["id"]]]
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
    names = {r["id"]: f"{ui.job_method_label(r)} · {ui.budget_label(r, results[r['id']])}"
                      f" · started {time.strftime('%H:%M:%S', time.localtime(r.get('created', 0)))}"
             for r in members}
    with st.container(border=True):
        a,b,c=st.columns(3)
        a.metric("Dataset",members[0].get("label") or members[0].get("dataset"))
        b.metric("Noise",f"{float(members[0].get('noise',0)):g}%")
        c.metric("Experiments",len(members))
        st.caption("Input fingerprint: "+key[:18]+"…" if not key.startswith("unverifiable:")
                   else "Numerical input fingerprint unavailable.")
    # Show different methods first, even when several EPDE retries are newer.
    defaults, seen_methods = [], set()
    for member in members:
        method = results[member["id"]].get("method")
        if method not in seen_methods and len(defaults) < 4:
            defaults.append(member["id"])
            seen_methods.add(method)
    defaults += [identifier for identifier in names if identifier not in defaults][:4-len(defaults)]
    picked=st.multiselect("Select runs",list(names),
                         default=defaults,max_selections=4,format_func=names.get)
    chosen=[(names[i],results[i]) for i in picked]
    if chosen:
        first=chosen[0][1]
        axes=ui.axes_of(first)
        analysis = analyze_comparison(chosen)
        insights_tab,equation_tab,metrics_tab,plots_tab,balance_tab=st.tabs(
            ["Insights","Equations","Metrics","Solution fit","Equation balance"])
        with insights_tab:
            st.subheader("What these runs show")
            st.caption("Each highlight answers a separate question about the selected runs. "
                       "A method can lead one metric and trail another.")
            for start in range(0, len(analysis["metrics"]), 2):
                cards = st.columns(2, gap="medium")
                for card, metric in zip(cards, analysis["metrics"][start:start+2]):
                    with card, st.container(border=True):
                        value = metric["value"]
                        formatted = ("—" if value is None else
                                     f"{value:.2f} s" if metric["key"] == "seconds" else
                                     f"{value:.4g}")
                        st.metric(metric["title"], formatted)
                        if metric["leaders"]:
                            heading = "Leads this metric" if len(metric["leaders"]) == 1 else "Tied on this metric"
                            st.markdown("**" + heading + "**")
                            for leader in metric["leaders"]:
                                st.success(leader)
                        else:
                            st.markdown("**No comparable leader**")
                        st.caption(metric["explanation"])
            for note in analysis["notes"]:
                st.caption(note)
            st.subheader("What to improve next")
            for (_, record), run in zip(chosen, analysis["runs"]):
                with st.container(border=True):
                    st.markdown("**" + ui.method_label(record) + "**")
                    st.caption(run["label"])
                    st.write(f"Status: {run['status']} · Selected equation: {run['selected_law']}")
                    for tip in run["advice"]:
                        st.write("• " + tip)
        with equation_tab:
            truth=(first.get("problem") or {}).get("truth")
            shared_truth = all((record.get("problem") or {}).get("truth") == truth
                               for _, record in chosen)
            if truth and shared_truth:
                st.markdown("**Known reference law**")
                ui.latex(truth,axes)
            # Four equations remain readable in a two-column grid.
            cards=[st.columns(min(2,len(chosen)),gap="medium") for _ in range((len(chosen)+1)//2)]
            for i,(name,record) in enumerate(chosen):
                card=cards[i//2][i%2]
                with card:
                    with st.container(border=True):
                        st.markdown("**"+ui.method_label(record)+"**")
                        st.caption(name)
                        if not shared_truth and (record.get("problem") or {}).get("truth"):
                            st.caption("Reference declared for this run")
                            ui.latex(record["problem"]["truth"], ui.axes_of(record))
                        if record.get("selected"):
                            ui.latex(record["selected"],ui.axes_of(record),record=record)
                        else:
                            st.info("No selected equation was saved.")
                        run = analysis["runs"][i]
                        verdict = (st.success if run["selected_law"] == "Exact structure" else
                                   st.warning if run["selected_law"] == "Different structure" else st.info)
                        verdict("Selected equation: " + run["selected_law"])
                        st.caption(f"Recorded search time: {run['seconds']:.1f} s"
                                   if run["seconds"] is not None else "Search time unavailable")
        with metrics_tab:
            methods_in_view={rec.get("method") for _,rec in chosen}
            summaries = []
            for (name, record), run in zip(chosen, analysis["runs"]):
                row = {"run": name, "method": ui.method_label(record), "status": run["status"],
                       "search budget": ui.budget_label({}, record),
                       "selected equation": run["selected_law"],
                       "HD": run["hd"], "RE (sum)": run["re"],
                       "solution R² (worst variable)": run["solution_r2"],
                       "candidates": len(record.get("front") or []),
                       "search time, s": run["seconds"]}
                if any(metric["key"] == "heldout_r2" for metric in analysis["metrics"]):
                    row["held-out forecast R²"] = run["heldout_r2"]
                summaries.append(row)
            st.dataframe(pd.DataFrame(summaries),hide_index=True,width="stretch",
                         column_config=ui.metric_columns())
            st.caption("Only matching input fingerprints are grouped. HD and RE follow the "
                       "PDEComp benchmark and are computed here in the same way for every method, "
                       "against the known law of this record. The methods' objectives and derivative "
                       "estimators differ; internal scores are not interchangeable. The Insights "
                       "tab checks which scores share an evaluation basis before highlighting leaders. "
                       "Solution R² is in-sample; unavailable scores remain blank. Coefficient errors "
                       "of an oracle best-front candidate are not scores of the selected equation.")
            for method in sorted(methods_in_view, key=str):
                st.caption("**"+ui.METHOD_NAMES.get(method, method)+"**: "+ui.PROTOCOL_NOTES.get(method, ""))
        with plots_tab:
            st.caption("Plots show solution reconstruction against the original data.")
            # Different boundary cuts can yield different observations/grids even
            # for one input. Only overlay reconstructions with the same basis.
            plot_groups = {}
            for index, (pair, run) in enumerate(zip(chosen, analysis["runs"])):
                basis = run["_solution_basis"] or ("unverified", index)
                plot_groups.setdefault(basis, []).append(pair)
            if len(plot_groups) > 1:
                st.caption("Separate plots keep each reconstruction on its own recorded grid. "
                           "Only runs with matching grids and observations are overlaid.")
            for group in plot_groups.values():
                if len(plot_groups) > 1:
                    st.markdown("**" + ", ".join(ui.method_label(record) for _, record in group) + "**")
                ui.solution_plots(group)
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
            incomplete = any(r["method"] == method and r["unscored runs"] for r in table)
            limit_label = ("not established" if incomplete else
                           "below the lowest level" if limit is None else f"{limit:g} %")
            col.metric(f"noise limit · {method}", limit_label,
                       help="Highest tested noise level up to which at least 3 of 5 runs (60 %) "
                            "found the law at every level, as in PDEComp. Needs at least 3 runs "
                            "per level.")
        if frame["unscored runs"].sum():
            st.info("Unscored results are included in the total run count. Success rates and "
                    "confidence intervals are unavailable for incomplete levels; the noise "
                    "limit is not established for methods with any unscored results.")
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
