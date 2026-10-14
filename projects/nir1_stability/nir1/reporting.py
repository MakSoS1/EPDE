"""Reviewer-facing Markdown generated solely from recorded observations."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence


def summarize_s0_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Collapse repeated candidate rows to one method × case × seed trial."""
    by_run: dict[tuple[object, object, object], dict[str, object]] = {}
    invalid_candidates = 0
    revisions = set()
    for row in rows:
        key = (row["case"], row["seed"], row["method"])
        compact = {"support_exact": bool(row["selected_support_exact"]),
                   "truth_first": row["truth_candidate_rank"] == 1,
                   "false_deleted": int(row["false_deleted"]),
                   "false_included": int(row["false_included"]),
                   "status": row["status"]}
        if key in by_run and by_run[key] != compact:
            raise ValueError(f"Conflicting candidate metrics for trial {key}")
        by_run[key] = compact
        invalid_candidates += row.get("score_status") != "scored"
        revisions.add(str(row["code_sha"]))
    by_method = defaultdict(list)
    for (_, _, method), result in by_run.items():
        by_method[str(method)].append(result)
    methods = {}
    for method, group in sorted(by_method.items()):
        methods[method] = {
            "n": len(group), "support_exact": sum(x["support_exact"] for x in group),
            "truth_first": sum(x["truth_first"] for x in group),
            "false_deleted": sum(x["false_deleted"] for x in group),
            "false_included": sum(x["false_included"] for x in group),
            "failures": sum(x["status"] != "ok" for x in group),
        }
    return {"stage": "S0", "planned": len(by_run), "ok": sum(
        x["status"] == "ok" for x in by_run.values()),
        "rows": len(rows), "invalid_candidate_scores": int(invalid_candidates),
        "methods": methods, "code_sha": next(iter(revisions)) if len(revisions) == 1 else "MIXED",
        "provisional": True, "full_epde_status": "NOT RUN",
        "historical_vclog_status": "ARCHIVAL; NOT REPRODUCED",
        "raw_data_status": "fixed candidates constructed from explicit simulator truth (S0 only)"}


def render_review(summary: Mapping[str, object], output_dir: Path) -> list[Path]:
    """Produce concise evidence/navigation docs; no missing-value substitution."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n, ok = int(summary.get("planned", 0)), int(summary.get("ok", 0))
    provenance = str(summary.get("code_sha", "NOT RECORDED"))
    methods = summary.get("methods", {})
    provisional = bool(summary.get("provisional", True))
    label = "PROVISIONAL" if provisional else "FROZEN"
    full = str(summary.get("full_epde_status", "NOT RUN"))
    table = ["| Method | S0 trials | Exact support | Truth candidate ranked first | False deletions | False inclusions |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, row in sorted(methods.items()):
        table.append(f"| `{name}` | {row['n']} | {row['support_exact']} | "
                     f"{row['truth_first']} | {row['false_deleted']} | {row['false_included']} |")
    if not methods:
        table.append("| NOT RUN | 0 | — | — | — | — |")
    summary_table = "\n".join(table)
    coverage = (f"{ok}/{n} S0 method-trials recorded" if n else "NOT RUN")
    explanation = (
        "S0 is a **fixed-candidate regression/ranking test** on controlled, "
        "synthetically generated X and y. A success here is **not** successful "
        "EPDE evolutionary discovery. Methods did not consume truth labels; "
        "truth was used only to construct the diagnostic candidate set and score results."
    )
    review = f"""# NIR-1 reviewer briefing — {label}

**Scope:** coefficient instability, identifiability diagnostics and adaptive regularization  
**Branch:** [`nir1`](https://github.com/MakSoS1/EPDE/tree/nir1)  
**Recorded local commit:** `{provenance}`  
**S0 coverage:** {coverage}  
**Full EPDE search:** **{full}**

## Verified from available evidence

{explanation}

{summary_table}

## What these results cannot establish

- No S1/S2 heldout full-search conclusion is available: **{full}**.
- Method counts share the same synthetic fixtures; they are not independent systems.
- `historical_proxy` does not reproduce the archive's 58/76 result.
- Ill-conditioned/weak-term failures must remain visible, not removed from the denominator.
- A comparison against current `chi2+vwsr` in real EPDE evolution is still required.

See [METHODS.md](METHODS.md), [RESULTS.md](RESULTS.md), and
`reports/s0/fixtures.jsonl.gz` (lossless gzip) for run-level verification.
"""
    results = f"""# NIR-1 findings — {label}

- **S0 candidate-level:** {coverage}; raw candidate rows: {summary.get('rows', 'NOT RUN')}.
- **Invalid/unscorable candidate scores:** {summary.get('invalid_candidate_scores', 'NOT RUN')}.
- **S1 full EPDE, S2 heldout, S3 transfer:** **{full}**.
- **Historical vclog+swap (58/76 vs 45/76):** archival, **not reproduced**.
- **Numerical improvement over EPDE baseline:** **NOT ESTABLISHED**.

{summary_table}

{explanation}

**Status:** {label}. Numerical rates must be qualified by full planned counts,
solver convergence and heldout split before making a scientific claim.
"""
    methods_doc = """# NIR-1 methods and limitations

1. T1–T8 controlled synthetic arrays with independent signal, target-noise and derivative-error RNG streams.
2. Weighted sufficient statistics: G=XᵀWX, b=XᵀWy, and exact support submatrix reuse.
3. Coefficient diagnostics: per-environment fitted-coefficient variance minus estimated iid uncertainty, scaled by normalized coefficient energy; SVD-based per-term identifiability Q.
4. Candidate-level OLS, LASSO, instability-weighted and identifiability-weighted elastic net. Tuning here uses fixed preregistered λ; it has not been optimized on heldout truth.
5. Fixed truth/decoy candidate sets are constructed by the simulation oracle, but data-only metrics and sparse fits read no truth. No EPDE evolution occurs in S0.
6. Historical vclog proxy is not the main-branch multi-axis correction; archive's 58/76 is an external reference only.
7. S1/S2 require whole-system heldout evaluation, 20 paired optimizer seeds, uncertainty estimates and recorded failures before confirming improvement.
8. E1 acceleration uses content-addressed Gram cache, parity checks and SVD/reference fallback. E2 approximate speedups are disabled by default.
9. Under errors in predictor derivatives, reported iid sampling variance is an **assumption check**, not calibrated confidence. `NOT RUN` is never zero accuracy.
"""
    russian = f"""# НИР-1 — кратко для научного руководителя

**Статус:** {label}. **Ветка:** `nir1`.

- Проверено на фиксированных синтетических кандидатах: {coverage}.
- Исследуются нестабильность коэффициентов, идентифицируемость, адаптивный L1/L2 штраф и коррелированные признаки.
- Результаты по методам и их ошибкам: см. `REVIEW.md` и `RESULTS.md`.
- Полноценный эволюционный поиск EPDE и подтверждение на отложенных задачах: **{full}**.
- Старый результат `58/76` нельзя считать результатом нового исследования без полного повторения метода.

Научно корректный следующий этап — полный поиск на фиксированных PIC-системах и парное сравнение с `chi2+vwsr`; нельзя пока заявлять улучшение качества EPDE.
"""
    contents = {"REVIEW.md": review, "RESULTS.md": results,
                "METHODS.md": methods_doc, "Кратко_для_ревью.md": russian}
    paths = []
    for name, content in contents.items():
        path = output_dir / name
        path.write_text(content, encoding="utf-8")
        paths.append(path)
    (output_dir / "summary.json").write_text(json.dumps(dict(summary), indent=2,
                                                         ensure_ascii=False, allow_nan=False), encoding="utf-8")
    paths.append(output_dir / "summary.json")
    return paths


def render_s0_figures(rows: Sequence[Mapping[str, object]], output_dir: Path) -> list[Path]:
    """Descriptive S0 graphs; full-search bars are never invented."""
    if not rows:
        return []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    from .fixtures import make_fixture
    from .diagnostics import estimate_diagnostics

    summary = summarize_s0_rows(rows)
    methods = summary["methods"]
    names = list(methods)
    n = [methods[k]["n"] for k in names]
    recovery = [100 * methods[k]["support_exact"] / max(methods[k]["n"], 1) for k in names]
    first = [100 * methods[k]["truth_first"] / max(methods[k]["n"], 1) for k in names]
    colors = ["#2563eb", "#0d9488", "#7c3aed", "#db2777", "#d97706", "#64748b"]
    outputs = []
    for index, (values, label, filename) in enumerate((
            (recovery, "Exact support in frozen candidate regression (%)", "01_support_recovery.png"),
            (first, "Known equation ranked first among frozen candidates (%)", "02_candidate_ranking.png"))):
        fig, ax = plt.subplots(figsize=(9.5, 4.8), layout="constrained")
        bar = ax.barh(names[::-1], values[::-1], color=[colors[i % len(colors)] for i in range(len(names))][::-1])
        ax.set_xlim(0, 108)
        ax.set_xlabel(label)
        ax.set_title("NIR-1 S0 | controlled synthetic arrays — not EPDE evolution")
        ax.grid(axis="x", alpha=.16)
        ax.set_axisbelow(True)
        for b in bar:
            ax.text(b.get_width() + 1, b.get_y() + b.get_height() / 2,
                    f"{b.get_width():.1f}%", va="center", fontsize=9)
        target = output_dir / filename
        fig.savefig(target, dpi=170)
        plt.close(fig)
        outputs.append(target)

    fig, ax = plt.subplots(figsize=(9.5, 4.8), layout="constrained")
    positions = list(range(len(names)))
    deleted = [methods[k]["false_deleted"] for k in names]
    included = [methods[k]["false_included"] for k in names]
    ax.barh(positions, deleted, color="#e11d48", label="False deleted terms")
    ax.barh(positions, included, left=deleted, color="#f59e0b", label="False included terms")
    ax.set_yticks(positions, names)
    ax.set_xlabel("Total term errors across synthetic runs (not normalized by term count)")
    ax.set_title("NIR-1 S0 | candidate-level support error audit")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=.15)
    target = output_dir / "03_term_errors.png"
    fig.savefig(target, dpi=170)
    plt.close(fig)
    outputs.append(target)

    corr = [0., .5, .9, .99, .999]
    q = []
    for rho in corr:
        design, _ = make_fixture("T2", seed=0, correlation=rho)
        diagnostic = estimate_diagnostics(design)
        q.append(float(diagnostic.identifiability[0]))
    fig, ax = plt.subplots(figsize=(8.2, 4.5), layout="constrained")
    ax.plot([str(x) for x in corr], q, marker="o", color="#2563eb", linewidth=2.4)
    ax.set_xlabel("Constructed correlation between columns")
    ax.set_ylabel("Identifiability Q of first term (0 = ambiguous)")
    ax.set_ylim(-.05, 1.05)
    ax.set_title("NIR-1 T2 | identifiability collapses for near-duplicate columns")
    ax.grid(alpha=.15)
    target = output_dir / "04_correlation_identifiability.png"
    fig.savefig(target, dpi=170)
    plt.close(fig)
    outputs.append(target)
    return outputs


def render_pilot_addendum(output_dir: Path, full_search_records: Sequence[Mapping[str, object]],
                          s0_ledger: Mapping[str, object] | None = None,
                          *, actions_runs: Sequence[Mapping[str, object]] = ()) -> list[Path]:
    """Truthful research appendix from raw EPDE runs and reconciled statuses.

    A full EPDE run with 1 seed is a feasibility pilot, not a statistically
    powered S1 comparison. Smoke records are explicitly excluded. No CI is
    generated until preregistered independent systems/seeds are available.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    full = [r for r in full_search_records if not r.get("nir1_smoke_only", False)]
    smoke = sum(bool(r.get("nir1_smoke_only", False)) for r in full_search_records)
    lines = ["# NIR-1 actual EPDE pilot — PROVISIONAL", "",
             "**Scope:** genuine PIC/EPDE evolutionary searches; NOT the S0 fixed-candidate tests.",
             "**Inferential status:** INSUFFICIENT independent systems/seeds for confidence intervals or superiority claims.",
             f"**SMOKE EXCLUDED:** {smoke} reduced-budget runs are not scientific trials.", ""]
    if s0_ledger:
        counts = s0_ledger["status_counts"]
        planned = int(s0_ledger["planned"])
        lines.extend(["## S0 frozen campaign reconciliation", "",
                      f"- **{counts['ok']}/{planned}** successful S0 method trials; "
                      f"**{counts['crash']} crashes**, {counts['timeout']} timeouts, "
                      f"{counts['unsupported']} unsupported, {counts['incomplete']} incomplete.",
                      "- Every planned identity is counted, including invalid/singular solver cases.", ""])
    lines.extend(["## Measured full-search records", "",
                  "The success column uses the **PIC predefined compromise selector** "
                  "(`metrics.success_selected`, truth-free min-max normalized objective sum). "
                  "A separate Euclidean research selector is recorded in the raw JSON, "
                  "but is not substituted after seeing truth. Source hashes here are "
                  "local Git revisions used at execution; GitHub-hosted mirrored commits "
                  "can have different commit IDs and must be matched by source contents.", "",
                  "| Dataset | Variant | Optimizer seed | Source revision | Status | PIC selected exact | Full fit (s) | Wall (s) |",
                  "|---|---|---:|---|---|---|---:|---:|"])
    for r in sorted(full, key=lambda x: (str(x.get("dataset")), int(x.get("seed", -1)),
                                         str(x.get("research_variant", "")))):
        status = str(r.get("status", "NOT RECORDED"))
        hit = r.get("metrics", {}).get("success_selected") if status == "ok" else None
        success = "yes" if hit is True else "no" if hit is False else "—"
        fit = r.get("fit_seconds")
        wall = r.get("total_seconds")
        fit_text = f"{float(fit):.1f}" if isinstance(fit, (int, float)) else "—"
        wall_text = f"{float(wall):.1f}" if isinstance(wall, (int, float)) else "—"
        revision = str(r.get("environment", {}).get("epde_commit", "UNRECORDED"))
        if "dirty" in revision:
            revision = f"{revision} (PRE-COMMIT; not frozen)"
        lines.append(f"| `{r.get('dataset', '?')}` | `{r.get('research_variant', '?')}` | "
                     f"{r.get('seed', '?')} | `{revision}` | {status} | {success} | "
                     f"{fit_text} | {wall_text} |")
    if not full:
        lines.append("| NOT RUN | — | — | — | — | — | — | — |")
    lines.extend(["", "## GitHub Actions evidence", ""])
    for run in actions_runs:
        zero_jobs = int(run.get("job_count", -1)) == 0
        verdict = ("NO JOBS; no VM experiment executed" if zero_jobs else
                   f"{run['job_count']} jobs; raw records independently reconciled"
                   if run.get("artifacts_verified") else
                   f"{run.get('job_count', '?')} job(s) observed; raw artifact verification required")
        lines.append(f"- [Run {run['run_id']}](https://github.com/MakSoS1/EPDE/actions/runs/{run['run_id']}): "
                     f"{run.get('conclusion', 'unknown')} — **{verdict}**.")
    if not actions_runs:
        lines.append("- No independently verified workflow results available.")
    lines.extend(["", "## Interpretation and next gate", "",
                  "This small, mostly one-seed multi-system engineering pilot is not proof of a method effect. "
                  "A slower or incorrect equation is a negative observation, not removed from statistics.",
                  "S2 heldout and S3 transfer cannot be advertised until the S1 finalist/budget are "
                  "frozen and Actions (or another authorized isolated runner) executes all planned IDs.",
                  "The historical 58/76 count is not reproduced here.", ""])
    target = output_dir / "PILOT.md"
    target.write_text("\n".join(lines), encoding="utf-8")
    for name in ("REVIEW.md", "RESULTS.md", "Кратко_для_ревью.md"):
        doc = output_dir / name
        if doc.exists():
            body = doc.read_text(encoding="utf-8")
            body += ("\n## Independently recorded EPDE pilot and S0 job statuses\n\n"
                     "The full-search observations, failed-run denominators and failed "
                     "GitHub Actions attempts are reported in [PILOT.md](PILOT.md). "
                     "The fixed-candidate S0 table above must not be presented as full EPDE accuracy.\n")
            doc.write_text(body, encoding="utf-8")
    return [target, *(output_dir / x for x in ("REVIEW.md", "RESULTS.md", "Кратко_для_ревью.md"))]
