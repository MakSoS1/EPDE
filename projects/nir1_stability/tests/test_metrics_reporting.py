from pathlib import Path
import gzip
import json
import subprocess
import sys

import numpy as np
import pytest

from projects.nir1_stability.nir1.metrics import (
    paired_effect_ci, summarize_outcomes, wilson_interval,
)
from projects.nir1_stability.nir1.reporting import (
    render_review, summarize_s0_rows, render_pilot_addendum,
)


def test_status_denominators_do_not_drop_crashes_timeouts_or_incomplete():
    ledger = {"planned": 10, "status_counts": {
        "ok": 5, "crash": 2, "timeout": 1, "unsupported": 1, "incomplete": 1}}
    summary = summarize_outcomes(ledger, [])
    assert summary["total_planned"] == 10
    assert summary["supported_planned"] == 9
    assert summary["supported_success_rate"] == 5 / 9
    assert summary["complete"] is False
    assert summary["status_counts"]["unsupported"] == 1


def test_empty_partial_summaries_do_not_contain_invented_improvement():
    summary = summarize_outcomes({"planned": 0, "status_counts": {
        "ok": 0, "crash": 0, "timeout": 0, "unsupported": 0, "incomplete": 0}}, [])
    assert summary["supported_success_rate"] is None
    assert summary["conclusion_status"] == "NOT RUN"


def test_paired_identical_results_have_zero_effect_and_ci():
    a = [True, False, True, True, False, True]
    effect = paired_effect_ci(a, a, ["ode", "ode", "vdp", "vdp", "ac", "ac"],
                              n_boot=300, seed=2)
    assert effect["delta_pp"] == 0
    assert effect["ci_low_pp"] == 0
    assert effect["ci_high_pp"] == 0
    assert effect["discordant"] == 0


def test_cluster_bootstrap_is_not_pointwise_bootstrap():
    a = [False, False, False, False, False, False]
    b = [True, True, False, False, True, True]
    result = paired_effect_ci(a, b, ["ode", "ode", "ac", "ac", "lv", "lv"],
                              n_boot=200, seed=0)
    assert result["n_clusters"] == 3
    assert result["delta_pp"] == pytest.approx(100 * 4 / 6, abs=1e-12)
    assert 0 <= result["ci_low_pp"] <= result["ci_high_pp"] <= 100


def test_wilson_interval_contains_observed_proportion():
    low, high = wilson_interval(5, 10)
    assert 0 < low < .5 < high < 1


def test_report_renders_even_if_campaign_not_run(tmp_path):
    summary = {"stage": "S0", "planned": 10, "ok": 0, "provisional": True,
               "code_sha": "abc", "methods": {}, "full_epde_status": "NOT RUN"}
    files = render_review(summary, tmp_path)
    assert {"REVIEW.md", "RESULTS.md", "METHODS.md", "Кратко_для_ревью.md"}.issubset(
        {p.name for p in files})
    text = (tmp_path / "RESULTS.md").read_text()
    assert "NOT RUN" in text
    assert "PROVISIONAL" in text


def test_s0_raw_rows_count_unique_method_runs_not_candidate_replicates():
    rows = [
        {"case": "T1", "seed": 0, "method": "lasso", "candidate": "truth", "status": "ok",
         "selected_support_exact": True, "truth_candidate_rank": 1, "false_deleted": 0,
         "false_included": 0, "code_sha": "abc", "score_status": "scored"},
        {"case": "T1", "seed": 0, "method": "lasso", "candidate": "extra_term_decoy", "status": "ok",
         "selected_support_exact": True, "truth_candidate_rank": 1, "false_deleted": 0,
         "false_included": 0, "code_sha": "abc", "score_status": "scored"},
    ]
    result = summarize_s0_rows(rows)
    assert result["planned"] == 1
    assert result["methods"]["lasso"]["n"] == 1
    assert result["methods"]["lasso"]["support_exact"] == 1


def test_cli_report_produces_review_without_claiming_full_epde(tmp_path):
    path = tmp_path / "s0.jsonl"
    row = {"case": "T1", "seed": 0, "method": "lasso", "candidate": "truth", "status": "ok",
           "selected_support_exact": False, "truth_candidate_rank": 2, "false_deleted": 1,
           "false_included": 0, "code_sha": "test", "score_status": "scored"}
    path.write_text(json.dumps(row) + "\n")
    out = tmp_path / "review"
    proc = subprocess.run([sys.executable, "-m", "projects.nir1_stability.nir1.cli",
                           "report", "--s0-results", str(path), "--output", str(out)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "NOT RUN" in (out / "REVIEW.md").read_text()


def test_lossless_compressed_raw_s0_evidence_can_be_reported(tmp_path):
    path = tmp_path / "s0.jsonl.gz"
    row = {"case": "T1", "seed": 0, "method": "lasso", "candidate": "truth", "status": "ok",
           "selected_support_exact": True, "truth_candidate_rank": 1,
           "false_deleted": 0, "false_included": 0,
           "code_sha": "test", "score_status": "scored"}
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")
    proc = subprocess.run([sys.executable, "-m", "projects.nir1_stability.nir1.cli",
                           "report", "--s0-results", str(path), "--output", str(tmp_path / "out")],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "1/1" in (tmp_path / "out/RESULTS.md").read_text()


def test_full_epde_pilot_addendum_never_hides_failure_or_counts_smoke(tmp_path):
    render_review({"stage": "S0", "planned": 2, "ok": 2, "methods": {},
                   "full_epde_status": "S1 PILOT ONLY", "code_sha": "abc"}, tmp_path)
    baseline = {"dataset": "ode", "seed": 0, "status": "ok", "research_variant": "default",
                "metrics": {"success_selected": True}, "fit_seconds": 60.,
                "total_seconds": 63., "front": [["true"]]}
    new = {**baseline, "research_variant": "nir1_combined",
           "metrics": {"success_selected": False}, "total_seconds": 180.,
           "fit_seconds": 170.}
    smoke = {**baseline, "nir1_smoke_only": True}
    ledger = {"planned": 10, "status_counts": {"ok": 8, "crash": 2,
              "timeout": 0, "unsupported": 0, "incomplete": 0}}
    files = render_pilot_addendum(tmp_path, [baseline, new, smoke], ledger,
                                  actions_runs=[{"run_id": 1, "conclusion": "failure", "job_count": 0}])
    assert (tmp_path / "PILOT.md") in files
    txt = (tmp_path / "PILOT.md").read_text()
    assert "8/10" in txt and "2 crashes" in txt
    assert "180.0" in txt and "63.0" in txt
    assert "SMOKE EXCLUDED" in txt
    assert "NO JOBS" in txt
    assert "INSUFFICIENT" in txt
