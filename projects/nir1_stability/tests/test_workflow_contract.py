from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]


def _load(filename):
    path = ROOT / ".github/workflows" / filename
    assert path.exists(), f"missing workflow: {path}"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    # PyYAML YAML 1.1 recognizes `on` as a boolean; GitHub does not.
    return data, (data.get("on") or data.get(True))


def test_smoke_is_read_only_and_scoped_to_nir1_branch():
    data, trigger = _load("nir1-smoke.yml")
    assert trigger["push"]["branches"] == ["nir1"]
    assert data["permissions"]["contents"] == "read"
    assert "pull_request" not in trigger
    assert "workflow_dispatch" not in trigger
    assert all(int(job["timeout-minutes"]) <= 360 for job in data["jobs"].values())


def test_heavy_workflow_runs_only_for_immutable_launch_manifest():
    data, trigger = _load("nir1-research.yml")
    assert trigger["push"]["branches"] == ["nir1"]
    paths = trigger["push"]["paths"]
    assert paths == ["projects/nir1_stability/manifests/launch/*.yaml"]
    assert data["permissions"]["contents"] == "read"
    assert "workflow_dispatch" not in trigger
    assert any("matrix" in str(job) for job in data["jobs"].values())
    assert all(int(job["timeout-minutes"]) <= 360 for job in data["jobs"].values())
    assert any("upload-artifact" in str(job) and "always()" in str(job)
               for job in data["jobs"].values())


def test_repo_workflows_do_not_upload_to_upstream_or_require_secrets():
    for name in ("nir1-smoke.yml", "nir1-research.yml"):
        path = ROOT / ".github/workflows" / name
        text = path.read_text(encoding="utf-8")
        assert "ITMO-NSS" not in text
        assert "secrets." not in text
        assert "contents: write" not in text
        assert "timeout-minutes: 340" in text if name == "nir1-research.yml" else "timeout-minutes: 30" in text
