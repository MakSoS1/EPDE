"""Failure-aware paired reporting from immutable Actions run records."""

import json
import subprocess
import sys
from pathlib import Path

from projects.nir1_stability.nir1.campaign import plan_campaign
from projects.nir1_stability.nir1.records import atomic_record
from projects.nir1_stability.nir1.s1_aggregate import analyze_s1, write_s1_report


def _manifest():
    return plan_campaign({
        "stage": "S1", "cases": ["ode", "vdp", "burgers"],
        "methods": ["default", "nir1_combined"], "data_seeds": [0],
        "optimizer_seeds": [0, 1], "shards": 4,
        "config_sha": "a" * 64, "code_sha": "b" * 40,
        "split_sha": "c" * 64, "estimated_minutes_per_run": 1,
    })


def _write(path: Path, manifest, run, status, *, hit=False, attempt=0):
    raw = {"run_id": run["run_id"], "manifest_sha": manifest["manifest_sha"],
           "status": status, "attempt": attempt, "elapsed_seconds": 2}
    if status == "ok":
        raw["rows"] = [{"status": "ok", "metrics": {"success_selected": hit,
                        "success_front": hit}, "total_seconds": 1.5}]
    elif status == "crash":
        raw["failure_kind"] = "algorithm"
    return atomic_record(path / f'{run["run_id"]}.json', raw)


def test_s1_counts_failures_as_false_and_blocks_inference_when_missing(tmp_path):
    manifest = _manifest()
    files = []
    for run in manifest["runs"][:-1]:
        status = "crash" if run["method"] == "nir1_combined" else "ok"
        files.append(_write(tmp_path, manifest, run, status, hit=True))
    result = analyze_s1(manifest, files)
    assert result["planned"] == 12
    assert result["methods"]["nir1_combined"]["planned"] == 6
    assert result["methods"]["nir1_combined"]["selected_exact"] == 0
    assert result["methods"]["nir1_combined"]["crash"] == 5
    assert result["methods"]["nir1_combined"]["incomplete"] == 1
    assert result["comparisons"]["nir1_combined"]["inference"] == "BLOCKED"
    assert result["comparisons"]["nir1_combined"]["unresolved_pairs"] == 1


def test_s1_full_paired_summary_and_holm_correction_are_reproducible(tmp_path):
    manifest = _manifest()
    files = []
    for run in manifest["runs"]:
        method = run["method"]
        files.append(_write(tmp_path, manifest, run,
                            "ok" if method == "default" else "crash",
                            hit=method == "default"))
    result = analyze_s1(manifest, files)
    contrast = result["comparisons"]["nir1_combined"]
    assert result["ledger"]["status_counts"]["crash"] == 6
    assert result["methods"]["default"]["selected_exact"] == 6
    assert contrast["inference"] == "RECORDED_DESCRIPTIVE_WITH_CI"
    assert contrast["n_clusters"] == 3
    assert contrast["delta_pp"] == -100
    assert contrast["holm_adjusted_p"] >= contrast["mcnemar_exact_p"]
    assert result == analyze_s1(manifest, files)
    targets = write_s1_report(tmp_path / "out", result)
    assert len(targets) == 2
    assert "failure" in (tmp_path / "out" / "S1_SUMMARY.md").read_text().lower()
    assert "-100" in (tmp_path / "out" / "S1_SUMMARY.md").read_text()


def test_cli_analyze_s1_produces_truthful_machine_and_review_artifacts(tmp_path):
    manifest = _manifest()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    for run in manifest["runs"]:
        _write(tmp_path / "raw", manifest, run, "ok", hit=run["method"] == "default")
    proc = subprocess.run([sys.executable, "-m", "projects.nir1_stability.nir1.cli",
                           "analyze-s1", "--manifest", str(tmp_path / "manifest.json"),
                           "--records", str(tmp_path / "raw"),
                           "--output", str(tmp_path / "report")],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["planned"] == 12
    result = json.loads((tmp_path / "report" / "S1_SUMMARY.json").read_text())
    assert result["inference_gate"] == "PROVISIONAL_S1_ONLY"
    assert result["methods"]["default"]["selected_exact"] == 6
