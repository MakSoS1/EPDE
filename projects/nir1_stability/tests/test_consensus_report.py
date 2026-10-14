"""Consensus report refuses partial/false evidence and accounts for cost."""

import json
from pathlib import Path

import pytest

from projects.nir1_stability.nir1.campaign import _manifest_sha
from projects.nir1_stability.nir1.consensus_report import analyze_consensus
from projects.nir1_stability.nir1.records import atomic_record


T = "2.0 * u{power: 1.0} + 0.0 = du/dx0{power: 1.0}"
F = "3.0 * u{power: 2.0} + 0.0 = du/dx0{power: 1.0}"


def _frozen_dataset(tmp_path: Path):
    records, runs = [], []
    for case in ["A", "B", "C"]:
        for seed in range(3):
            rid = f"{len(runs)+1:064x}"
            idx = (seed > 0)
            winner = T if idx else F
            payload = {
                "run_id": rid, "status": "ok",
                "manifest_sha": "filled below", "attempt": 0,
                "rows": [{
                    "status": "ok", "research_variant": "default",
                    "dataset": case, "seed": seed,
                    "noise": 0, "search_config": {"case": case},
                    "config": {"nir1_data_seed": 0},
                    "environment": {"epde_commit": "1234567"},
                    "front": [[winner]], "metrics": {
                        "selected_index": 0, "success_selected": bool(idx)},
                    "total_seconds": 10.0,
                }],
            }
            records.append(payload)
            runs.append({"run_id": rid, "case": case, "method": "default",
                         "data_seed": 0, "optimizer_seed": seed, "shard": str(seed)})
    manifest = {
        "stage": "S2", "code_sha": "123456789abcdef",
        "runs": runs, "max_infrastructure_retries": 2,
    }
    manifest["manifest_sha"] = _manifest_sha(manifest)
    paths = []
    for row in records:
        row["manifest_sha"] = manifest["manifest_sha"]
        path = tmp_path / (row["run_id"] + ".json")
        atomic_record(path, row)
        paths.append(path)
    return manifest, paths


def test_complete_report_uses_one_system_as_one_statistical_unit(tmp_path):
    manifest, files = _frozen_dataset(tmp_path)
    result = analyze_consensus(manifest, files)
    assert result["n_independent_systems"] == 3
    assert result["n_total_full_searches"] == 9
    assert result["mean_native_single_search_success"] == pytest.approx(2/3)
    assert result["consensus_system_success"] == pytest.approx(1.)
    assert result["difference_pp_descriptive"] == pytest.approx(100/3)
    assert result["exact_system_cluster_signflip_p"] == pytest.approx(.25)
    assert all(x["total_cpu_wall_seconds"] == pytest.approx(30.) for x in result["systems"])


def test_missing_full_search_blocks_inference(tmp_path):
    manifest, paths = _frozen_dataset(tmp_path)
    result = analyze_consensus(manifest, paths[:-1])
    assert result["status"] == "INCOMPLETE_OR_FAILURE_BLOCKS_CONFIRMATION"
    assert "difference_pp_descriptive" not in result
    assert result["ledger"]["status_counts"]["incomplete"] == 1


def test_bad_digest_reported_not_hidden(tmp_path):
    manifest, paths = _frozen_dataset(tmp_path)
    p = paths[-1]
    changed = json.loads(p.read_text())
    changed["status"] = "crash"
    p.write_text(json.dumps(changed))
    result = analyze_consensus(manifest, paths)
    assert result["status"] == "INCOMPLETE_OR_FAILURE_BLOCKS_CONFIRMATION"
    assert result["ledger"]["invalid_records"]
