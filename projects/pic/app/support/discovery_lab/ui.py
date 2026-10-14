"""Rendering helpers for the Discovery Lab: one result, and several results side by side.

Nothing here runs a search; everything reads the JSON written by the worker
(``evaluation`` holds the thinned solution series, so pages stay fast).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from ..common import show_figure
from ..equations import system_latex, term_plain

METHOD_NAMES = {"epde": "EPDE", "pysindy": "PySINDy", "llm": "LLM agent"}
METHOD_COLORS = {"epde": "#4c78a8", "pysindy": "#f58518", "llm": "#54a24b"}
EFFORT_NAMES = {"quick": "quick", "standard": "standard", "thorough": "thorough"}


def axes_of(record):
    return ((record.get("evaluation") or {}).get("axis_names")
            or (record.get("problem") or {}).get("axis_names") or ["t", "x", "y"])


def latex(system, axes):
    try:
        st.latex(system_latex(system, axes))
    except Exception:                                   # noqa: BLE001
        st.code("\n".join(system), language=None)


def verdict(record):
    """(kind, text) against the known law, or None when no law is known."""
    truth = (record.get("problem") or {}).get("truth")
    if not truth:
        return None
    m = record.get("metrics") or {}
    if m.get("success_selected"):
        text = "Found the known law: the selected equation has exactly its terms."
        if m.get("coef_error") is not None:
            text += f" Mean coefficient error {m['coef_error']:.2%}."
        return "success", text
    if m.get("success_front"):
        return "warning", ("The known law is among the candidates, but another equation was "
                           "selected. See *All candidates*.")
    if m.get("hamming_min") is not None:
        return "error", (f"The known law was not found: the closest candidate differs by "
                         f"{m['hamming_min']} term(s).")
    return "error", "The known law was not found."


def solution_scores(record):
    """{variable: (R², normalized RMSE)} of the integrated equation, or {}."""
    solution = (record.get("evaluation") or {}).get("solution") or {}
    return {var: (solution.get("r2", {}).get(var), solution.get("nrmse", {}).get(var))
            for var in solution.get("r2", {})}


def balance_score(record):
    values = [v for v in ((record.get("evaluation") or {}).get("balance") or {}).values()
              if v is not None]
    return min(values) if values else None


def _fmt(value, pattern="{:.4f}"):
    return "—" if value is None else pattern.format(value)


def summary_row(job, record):
    """One descriptive row for tables (never a score combining methods)."""
    m = record.get("metrics") or {}
    truth = (record.get("problem") or {}).get("truth")
    scores = solution_scores(record)
    row = {"method": METHOD_NAMES.get(record.get("method"), record.get("method")),
           "search budget": EFFORT_NAMES.get(record.get("effort") or job.get("effort"), "standard"),
           "found equation": " ; ".join(plain_equation(eq, axes_of(record))
                                        for eq in (record.get("selected") or [])),
           "known law found": (("yes" if m.get("success_selected") else
                                "on the front" if m.get("success_front") else "no") if truth else "law unknown"),
           "coefficient error": m.get("coef_error") if truth and m.get("success_selected") else None,
           "solution R² (worst variable)": min((s[0] for s in scores.values() if s[0] is not None),
                                               default=None),
           "equation balance R²": balance_score(record),
           "candidates": len(record.get("front") or []),
           "search time, s": record.get("fit_seconds")}
    return row


def plain_equation(eq, axes):
    """``u_t = −0.5·u + sin(t)``: a short readable form for tables."""
    return f"{term_plain(eq.split('=', 1)[1], axes)} = {_rhs_plain(eq, axes)}"


def _rhs_plain(eq, axes):
    lhs, rhs = eq.split("=", 1)
    parts = []
    for chunk in lhs.split(" + "):
        chunk = chunk.strip()
        if not chunk:
            continue
        coef, _, body = chunk.partition(" * ")
        try:
            value = float(coef)
        except ValueError:
            parts.append(term_plain(chunk, axes))
            continue
        if not body:
            if value:
                parts.append(f"{value:.3g}")
            continue
        parts.append(f"{value:.3g}·{term_plain(body, axes)}")
    return " + ".join(parts).replace("+ -", "− ") or "0"


def term_table(record):
    from ..compare import match_equations, term_rows
    truth = (record.get("problem") or {}).get("truth")
    found = record.get("selected") or []
    if not truth or not found:
        return
    axes = axes_of(record)
    rows = []
    for truth_eq, found_eq in match_equations(found, truth):
        for row in term_rows(found_eq, truth_eq, axes):
            if len(truth) > 1:
                row = {"equation for": term_plain(truth_eq.split("=")[1], axes), **row}
            rows.append(row)
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def pareto(record):
    import matplotlib.pyplot as plt
    labels = record.get("criterion_labels") or ["objective 1", "objective 2"]
    objectives = record.get("objectives") or []
    selected = (record.get("metrics") or {}).get("selected_index")
    points = [(i, o) for i, o in enumerate(objectives) if o]
    if not points:
        st.caption("This method reported no two-criterion values.")
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    color = METHOD_COLORS.get(record.get("method"), "C0")
    ax.scatter([o[0] for _, o in points], [o[1] for _, o in points], s=36, color=color, zorder=2)
    for i, o in points:
        ax.annotate(str(i), (o[0], o[1]), textcoords="offset points", xytext=(4, 3), fontsize=8)
        if i == selected:
            ax.scatter([o[0]], [o[1]], s=150, facecolors="none", edgecolors="#e45756", lw=2,
                       zorder=3, label="selected equation")
    if selected is not None:
        ax.legend(fontsize=8)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.grid(alpha=.3)
    fig.tight_layout()
    show_figure(fig)
    if len(points) == 1:
        st.caption("One candidate only: a point, not a front.")


def _series(values):
    return np.array([np.nan if v is None else v for v in values], dtype=float)


def solution_plots(records):
    """Data against the solution of each record's equation, overlaid per variable.

    ``records`` is a list of (title, record) on the same input."""
    import matplotlib.pyplot as plt
    first = (records[0][1].get("evaluation") or {})
    kind = first.get("kind", "")
    solved = [(title, r) for title, r in records
              if ((r.get("evaluation") or {}).get("solution") or {}).get("observed")]
    if not solved:
        notes = {(r.get("evaluation") or {}).get("solution_note") for _, r in records} - {None}
        st.info(" ".join(notes) or "No solution of the equation could be computed for this input.")
        return
    reference = solved[0][1]["evaluation"]["solution"]
    variables = list(reference["observed"])
    if kind.startswith("ode"):
        t = _series(reference["t"])
        fig, axes = plt.subplots(len(variables), 1, figsize=(10, 2.6 * len(variables)),
                                 squeeze=False, sharex=True)
        for ax, var in zip(axes[:, 0], variables):
            ax.plot(t, _series(reference["observed"][var]), color="black", lw=1.6, label="data")
            for title, record in solved:
                sol = record["evaluation"]["solution"]
                if var in sol.get("simulated", {}):
                    ax.plot(_series(sol["t"]), _series(sol["simulated"][var]), "--", lw=1.3,
                            color=METHOD_COLORS.get(record.get("method")), label=title)
            data = _series(reference["observed"][var])
            low, high = np.nanmin(data), np.nanmax(data)
            margin = 0.5 * ((high - low) or 1.0)
            ax.set_ylim(low - margin, high + margin)      # a diverging solution stays readable
            ax.set_ylabel(var)
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(alpha=.25)
        axes[-1, 0].set_xlabel(axes_of(solved[0][1])[0])
        fig.tight_layout()
        show_figure(fig)
        return
    # 1-D PDE: data, each solution and its difference on one colour scale per variable
    for var in variables:
        observed = np.array(reference["observed"][var], dtype=float)
        lim = np.nanpercentile(np.abs(observed), 99) or 1.0
        t, x = _series(reference["t"]), _series(reference["x"])
        extent = [t[0], t[-1], x[0], x[-1]]
        rows = len(solved)
        fig, axes = plt.subplots(rows + 1, 2, figsize=(11, 2.6 * (rows + 1)), squeeze=False)
        im = axes[0, 0].imshow(observed.T, origin="lower", aspect="auto", cmap="RdBu_r",
                               vmin=-lim, vmax=lim, extent=extent)
        axes[0, 0].set_title(f"{var}: data", fontsize=9)
        fig.colorbar(im, ax=axes[0, 0])
        axes[0, 1].axis("off")
        for row, (title, record) in enumerate(solved, start=1):
            sim = np.array(record["evaluation"]["solution"]["simulated"][var], dtype=float)
            for ax, field, name, cmap in [(axes[row, 0], sim, "solution of the equation", "RdBu_r"),
                                          (axes[row, 1], sim - observed, "solution − data", "PuOr_r")]:
                im = ax.imshow(field.T, origin="lower", aspect="auto", cmap=cmap,
                               vmin=-lim, vmax=lim, extent=extent)
                ax.set_title(f"{title}: {name}", fontsize=9)
                fig.colorbar(im, ax=ax)
        for ax in axes.ravel():
            if ax.has_data():
                ax.set_xlabel("t")
                ax.set_ylabel("x")
        fig.tight_layout()
        show_figure(fig)
