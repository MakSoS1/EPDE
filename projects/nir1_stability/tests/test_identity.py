"""Reproducible NIR-1 run identity, independent of source-key order."""

from pathlib import Path

import pytest

from projects.nir1_stability.nir1.identity import (
    canonical_run_id,
    hash_file,
    verify_git_revision,
)


def _spec():
    return {
        "dataset": "burgers", "variant": "default", "noise": 0.01,
        "data_seed": 4, "optimizer_seed": 7, "config_sha": "a" * 64,
        "code_sha": "b" * 40, "split_sha": "c" * 64,
    }


def test_run_id_is_key_order_independent():
    source = _spec()
    assert canonical_run_id(source) == canonical_run_id(dict(reversed(list(source.items()))))


@pytest.mark.parametrize("field,value", [
    ("dataset", "ac"), ("data_seed", 5), ("optimizer_seed", 0),
    ("config_sha", "d" * 64), ("code_sha", "e" * 40),
    ("split_sha", "f" * 64),
])
def test_significant_change_changes_id(field, value):
    changed = _spec()
    changed[field] = value
    assert canonical_run_id(changed) != canonical_run_id(_spec())


def test_missing_identity_field_fails():
    incomplete = _spec()
    del incomplete["code_sha"]
    with pytest.raises(ValueError, match="code_sha"):
        canonical_run_id(incomplete)


def test_hash_file_rejects_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        hash_file(tmp_path / "missing")


def test_verify_git_revision_reads_pinned_base():
    repo = Path(__file__).resolve().parents[3]
    with pytest.raises(ValueError, match="revision"):
        verify_git_revision("0" * 40, repo)


def test_research_documents_present():
    repo = Path(__file__).resolve().parents[3]
    assert (repo / "docs/research/NIR1_EPDE_Research_Protocol_2026-10-10.md").is_file()
    assert (repo / "docs/superpowers/specs/2026-10-10-epde-nir1-implementation-design.md").is_file()
