"""Atomic checksummed run artifacts and corruption detection."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Mapping


STATUSES = ("ok", "timeout", "crash", "unsupported", "incomplete")


def canonical_bytes(row: Mapping[str, object]) -> bytes:
    return json.dumps(dict(row), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def atomic_record(path: Path, record: Mapping[str, object]) -> Path:
    path = Path(path)
    body = dict(record)
    if body.get("status") not in STATUSES:
        raise ValueError("Unknown run status")
    body.pop("record_sha256", None)
    body["record_sha256"] = hashlib.sha256(canonical_bytes(body)).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp: str | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", prefix=".tmp-nir1-", dir=path.parent,
                                         delete=False) as stream:
            temp = stream.name
            stream.write(canonical_bytes(body))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if temp and os.path.exists(temp):
            os.unlink(temp)
    return path


def read_record(path: Path) -> dict[str, object]:
    body = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = body.pop("record_sha256", None)
    actual = hashlib.sha256(canonical_bytes(body)).hexdigest()
    if actual != expected:
        raise ValueError(f"Record checksum mismatch: {path}")
    if body.get("status") not in STATUSES:
        raise ValueError(f"Invalid record status: {path}")
    body["record_sha256"] = expected
    return body
