import json
import subprocess
import sys

import pytest

from projects.nir1_stability.nir1.campaign import (
    execute_shard, plan_campaign, resume_plan, restore_shard_records,
)
from projects.nir1_stability.nir1.records import atomic_record, read_record


def config():
    return {"stage": "S0", "cases": ["T1"], "methods": ["lasso"],
            "data_seeds": [0, 1, 2, 3, 4], "optimizer_seeds": [0],
            "config_sha": "a" * 64, "code_sha": "b" * 40,
            "split_sha": "c" * 64, "shards": 1,
            "estimated_minutes_per_run": 2}


def test_resume_returns_only_three_missing_after_two_completed(tmp_path):
    manifest = plan_campaign(config())
    paths = []
    for run in manifest["runs"][:2]:
        paths.append(atomic_record(tmp_path / f'{run["run_id"]}.json',
                                   {"run_id": run["run_id"], "status": "ok", "rows": []}))
    summary = resume_plan(manifest, paths)
    assert summary["planned"] == 5
    assert summary["status_counts"]["ok"] == 2
    assert summary["status_counts"]["incomplete"] == 3
    assert set(summary["remaining"]) == {run["run_id"] for run in manifest["runs"][2:]}


def test_conflicting_duplicate_record_rejected(tmp_path):
    manifest = plan_campaign(config())
    rid = manifest["runs"][0]["run_id"]
    a = atomic_record(tmp_path / "a.json", {"run_id": rid, "status": "ok", "rows": []})
    b = atomic_record(tmp_path / "b.json", {"run_id": rid, "status": "ok", "rows": [1]})
    with pytest.raises(ValueError, match="conflict"):
        resume_plan(manifest, [a, b])


def test_corrupted_file_never_counts_as_success(tmp_path):
    manifest = plan_campaign(config())
    run = manifest["runs"][0]
    path = atomic_record(tmp_path / "bad.json", {"run_id": run["run_id"], "status": "ok", "rows": []})
    path.write_text(path.read_text().replace('"ok"', '"ab"'))
    with pytest.raises(ValueError, match="checksum"):
        read_record(path)
    summary = resume_plan(manifest, [path])
    assert summary["status_counts"]["ok"] == 0
    assert summary["invalid_records"]


def test_seven_hour_shard_is_rejected():
    c = config()
    c["estimated_minutes_per_run"] = 90
    with pytest.raises(ValueError, match="soft deadline"):
        plan_campaign(c)


def test_bounded_retry_and_algorithm_crash_are_not_laundered(tmp_path):
    manifest = plan_campaign(config())
    run = manifest["runs"][0]
    path = atomic_record(tmp_path / "crash.json", {"run_id": run["run_id"],
                                                      "status": "crash", "failure_kind": "algorithm"})
    summary = resume_plan(manifest, [path])
    assert run["run_id"] not in summary["remaining"]
    assert summary["status_counts"]["crash"] == 1


def test_shard_atomic_execution_skips_matching_completed_run(tmp_path):
    c = config()
    c["data_seeds"] = [0, 1]
    manifest = plan_campaign(c)
    path = tmp_path / "planned.json"
    path.write_text(json.dumps(manifest))
    assert execute_shard(path, "0", tmp_path / "runs") == 0
    first = [read_record(x) for x in sorted((tmp_path / "runs").glob("*.json"))]
    assert len(first) == 2
    assert all(row["status"] == "ok" for row in first)
    assert execute_shard(path, "0", tmp_path / "runs") == 0
    second = [read_record(x) for x in sorted((tmp_path / "runs").glob("*.json"))]
    assert first == second


def test_cli_plan_and_summarize_reconcile_partial_runs(tmp_path):
    launch = tmp_path / "s0.yaml"
    launch.write_text("stage: S0\ncases: [T1]\nmethods: [lasso]\n"
                      "data_seeds: [0, 1]\noptimizer_seeds: [0]\n"
                      "shards: 4\nestimated_minutes_per_run: 2\n")
    planned = tmp_path / "planned.json"
    cmd = [sys.executable, "-m", "projects.nir1_stability.nir1.cli"]
    done = subprocess.run(cmd + ["plan", "--launch", str(launch), "--output", str(planned)],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    m = json.loads(planned.read_text())
    assert len(m["runs"]) == 2
    output = tmp_path / "summary.json"
    done = subprocess.run(cmd + ["summarize", "--manifest", str(planned),
                                 "--records", str(tmp_path / "missing"),
                                 "--output", str(output)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    summary = json.loads(output.read_text())
    assert summary["planned"] == 2
    assert summary["status_counts"]["incomplete"] == 2


def test_retried_infrastructure_timeout_is_preserved_and_eventually_successful(tmp_path):
    manifest = plan_campaign(config())
    rid = manifest["runs"][0]["run_id"]
    first = atomic_record(tmp_path / "try0.json", {"run_id": rid, "status": "timeout",
                                                   "attempt": 0, "failure_kind": "infrastructure"})
    second = atomic_record(tmp_path / "try1.json", {"run_id": rid, "status": "ok",
                                                    "attempt": 1, "rows": []})
    result = resume_plan(manifest, [first, second])
    assert result["status_counts"]["ok"] == 1
    assert result["retry_count"] == 1
    assert result["attempt_history"][rid] == ["timeout", "ok"]


def test_restore_only_records_for_own_shard_and_preserves_checksums(tmp_path):
    manifest = plan_campaign(config())
    source = tmp_path / "past" / "attempt0"
    copied = atomic_record(source / f'{manifest["runs"][0]["run_id"]}.json',
                           {"run_id": manifest["runs"][0]["run_id"], "status": "ok", "rows": []})
    restored = restore_shard_records(manifest, "0", tmp_path / "past", tmp_path / "next")
    assert restored == 1
    assert read_record(tmp_path / "next" / copied.name) == read_record(copied)


def test_s1_shard_runs_actual_pic_adapter_instead_of_unsupported_stub(tmp_path, monkeypatch):
    from projects.nir1_stability.nir1 import campaign
    called = []

    def fake_subprocess(argv, **kwargs):
        called.append(argv)
        path = Path(argv[argv.index("--output") + 1])
        path.write_text(json.dumps({"status": "ok", "front": [["u_t=u"]],
                                    "objectives": [[1., 2.]],
                                    "metrics": {"selected_exact": True}}))
        return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

    from pathlib import Path
    monkeypatch.setattr(campaign.subprocess, "run", fake_subprocess)
    c = config()
    c.update(stage="S1", cases=["ode"], methods=["nir1_combined"],
             data_seeds=[2], optimizer_seeds=[7], noise=.01,
             search_overrides={"search": {"evolution": {"training_epochs": 2}}})
    m = plan_campaign(c)
    manifest_path = tmp_path / "m.json"
    manifest_path.write_text(json.dumps(m))
    assert execute_shard(manifest_path, "0", tmp_path / "runs") == 0
    records = list((tmp_path / "runs").glob("*.json"))
    assert len(records) == 1
    result = read_record(records[0])
    assert result["status"] == "ok"
    assert result["rows"][0]["front"] == [["u_t=u"]]
    cmd = called[0]
    assert cmd[cmd.index("--dataset") + 1] == "ode"
    assert cmd[cmd.index("--variant") + 1] == "nir1_combined"
    assert cmd[cmd.index("--seed") + 1] == "7"
    assert cmd[cmd.index("--data-seed") + 1] == "2"
    assert cmd[cmd.index("--noise") + 1] == "0.01"
