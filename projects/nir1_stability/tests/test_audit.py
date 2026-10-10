import json
import subprocess
import sys

from projects.nir1_stability.nir1.audit import evaluate_candidate_set, score_fixed_candidate
from projects.nir1_stability.nir1.design import ResearchDesign
from projects.nir1_stability.nir1.fixtures import make_fixture


def test_fixed_candidate_scores_do_not_read_truth_labels():
    design, _ = make_fixture("T1", seed=3)
    modified = ResearchDesign(design.X, design.y, design.sample_weight,
                              design.environment_id, design.token_names,
                              {**design.metadata, "truth": "completely wrong"})
    assert score_fixed_candidate(design) == score_fixed_candidate(modified)


def test_distinct_criterion_families_are_not_silently_same_scorer():
    design, _ = make_fixture("T4", seed=4)
    residual_score = score_fixed_candidate(design, criterion="residual")
    stability_score = score_fixed_candidate(design, criterion="nir1_excess")
    assert residual_score != stability_score
    assert residual_score >= 0 and stability_score >= 0


def test_audit_reports_distinct_candidate_ranking_and_support_metrics():
    rows = evaluate_candidate_set("T1", ["baseline_ols", "identifiability"], seed=4)
    assert len(rows) >= 4
    assert len({row["run_id"] for row in rows}) == len(rows)
    assert any(row["candidate"] == "truth" for row in rows)
    assert all("truth_candidate_rank" in row for row in rows)
    assert all("selected_support_exact" in row for row in rows)
    assert all(row["source_stage"] == "fixed_candidate" for row in rows)
    assert all(row["selected_by_search"] is False for row in rows)


def test_repeated_seed_produces_same_run_id_and_model_support():
    a = evaluate_candidate_set("T2", ["lasso"], seed=9)
    b = evaluate_candidate_set("T2", ["lasso"], seed=9)
    assert [(r["run_id"], r["selected_support"]) for r in a] == [
        (r["run_id"], r["selected_support"]) for r in b]


def test_cli_writes_valid_jsonl_with_traceable_provenance(tmp_path):
    output = tmp_path / "s0.jsonl"
    cmd = [sys.executable, "-m", "projects.nir1_stability.nir1.cli", "audit",
           "--cases", "T1,T2", "--seeds", "0-1", "--methods", "baseline_ols,lasso",
           "--output", str(output)]
    done = subprocess.run(cmd, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) >= 8
    assert all("code_sha" in row and "dataset_sha" in row for row in rows)
    assert all(row["status"] == "ok" for row in rows)
