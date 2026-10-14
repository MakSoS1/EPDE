"""Content-derived identities for reproducible, resumable experiments."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Mapping


REQUIRED_FIELDS = (
    "dataset", "variant", "noise", "data_seed", "optimizer_seed",
    "config_sha", "code_sha", "split_sha",
)


def _reject_nonfinite(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Run identities may not contain non-finite numbers")
    if isinstance(value, dict):
        for part in value.values():
            _reject_nonfinite(part)
    if isinstance(value, (list, tuple)):
        for part in value:
            _reject_nonfinite(part)


def canonical_run_id(spec: Mapping[str, object]) -> str:
    """Hash *all* immutable inputs; do not normalize away additional fields."""
    missing = [key for key in REQUIRED_FIELDS if key not in spec]
    if missing:
        raise ValueError(f"Missing run identity fields: {', '.join(missing)}")
    data = dict(spec)
    _reject_nonfinite(data)
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"),
                         allow_nan=False, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_git_revision(expected_sha: str, repo_root: Path) -> None:
    """Require an exact pinned base; never silently rebase onto new work."""
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True,
    ).strip()
    if actual != expected_sha:
        raise ValueError(f"Git revision mismatch: expected {expected_sha}, got {actual}")
