"""Immutable, resumable CPU research campaign for isolated Actions shards."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Mapping, Sequence

from .identity import canonical_run_id
from .records import STATUSES, atomic_record, canonical_bytes, read_record


def _manifest_sha(plan: Mapping[str, object]) -> str:
    filtered = {k: v for k, v in plan.items() if k != "manifest_sha"}
    return hashlib.sha256(canonical_bytes(filtered)).hexdigest()


def plan_campaign(config: Mapping[str, object]) -> dict[str, object]:
    """Expand a frozen config to individual run IDs with a bounded shard cap."""
    stage = str(config["stage"])
    if stage not in {"S0", "S1", "S2", "S3"}:
        raise ValueError("Unknown campaign stage")
    num_shards = int(config.get("shards", 1))
    est = float(config.get("estimated_minutes_per_run", 1.))
    soft = int(config.get("soft_deadline_minutes", 300))
    hard = int(config.get("hard_deadline_minutes", 340))
    if num_shards < 1 or est <= 0 or not 0 < soft < hard <= 360:
        raise ValueError("Invalid shard resource limits")
    combos = [(case, method, int(data_seed), int(opt_seed))
              for case in config["cases"] for method in config["methods"]
              for data_seed in config["data_seeds"]
              for opt_seed in config.get("optimizer_seeds", [0])]
    if est * math.ceil(len(combos) / num_shards) > soft:
        raise ValueError("Estimated shard runtime exceeds soft deadline")
    runs = []
    for i, (case, method, data_seed, optimizer_seed) in enumerate(combos):
        spec = {"dataset": case, "variant": method, "noise": config.get("noise", 0.),
                "data_seed": data_seed, "optimizer_seed": optimizer_seed,
                "config_sha": str(config["config_sha"]),
                "code_sha": str(config["code_sha"]),
                "split_sha": str(config["split_sha"]), "stage": stage}
        runs.append({"run_id": canonical_run_id(spec), "case": case, "method": method,
                     "data_seed": data_seed, "optimizer_seed": optimizer_seed,
                     "shard": str(i % num_shards)})
    if len({r["run_id"] for r in runs}) != len(runs):
        raise ValueError("Duplicate run identity; check frozen config")
    plan = {"stage": stage, "runs": runs, "shards": num_shards,
            "config_sha": str(config["config_sha"]), "code_sha": str(config["code_sha"]),
            "split_sha": str(config["split_sha"]),
            "noise": float(config.get("noise", 0.)),
            "search_overrides": dict(config.get("search_overrides", {})),
            "max_run_minutes": min(float(config.get("max_run_minutes", 120)), soft),
            "frozen_confirmation_sha": config.get("frozen_confirmation_sha"),
            "soft_deadline_minutes": soft, "hard_deadline_minutes": hard,
            "max_infrastructure_retries": min(int(config.get("max_infrastructure_retries", 2)), 2)}
    plan["manifest_sha"] = _manifest_sha(plan)
    return plan


def resume_plan(ledger_path: Mapping[str, object] | Path,
                available_records: Sequence[Path]) -> dict[str, object]:
    """Never discard failures or silently prefer a favorable duplicate."""
    manifest = (json.loads(Path(ledger_path).read_text(encoding="utf-8"))
                if isinstance(ledger_path, Path) else dict(ledger_path))
    if manifest.get("manifest_sha") != _manifest_sha(manifest):
        raise ValueError("Manifest checksum mismatch; immutable plan was edited")
    planned = {run["run_id"]: run for run in manifest["runs"]}
    seen: dict[str, dict[int, dict[str, object]]] = {}
    invalid: list[str] = []
    for path in available_records:
        try:
            record = read_record(path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            invalid.append(f"{path}: {error}")
            continue
        rid = str(record["run_id"])
        if rid not in planned:
            raise ValueError(f"Run {rid} not in immutable manifest")
        if record.get("manifest_sha", manifest["manifest_sha"]) != manifest["manifest_sha"]:
            raise ValueError(f"Run {rid} belongs to a different manifest")
        attempt = int(record.get("attempt", 0))
        if not 0 <= attempt <= int(manifest.get("max_infrastructure_retries", 2)):
            raise ValueError(f"Invalid attempt index for {rid}: {attempt}")
        attempts = seen.setdefault(rid, {})
        if attempt in attempts and attempts[attempt] != record:
            raise ValueError(f"conflicting duplicate record: {rid}, attempt {attempt}")
        attempts[attempt] = record
    counts = {status: 0 for status in STATUSES}
    remaining = []
    retry_limit = int(manifest.get("max_infrastructure_retries", 2))
    history: dict[str, list[str]] = {}
    for rid in planned:
        attempts = seen.get(rid, {})
        ordered = sorted(attempts)
        if ordered and ordered != list(range(ordered[-1] + 1)):
            raise ValueError(f"Missing earlier retry attempt for {rid}")
        for index in ordered[1:]:
            earlier = attempts[index - 1]
            if earlier["status"] not in {"timeout", "incomplete", "crash"} or (
                    earlier.get("failure_kind", "infrastructure") != "infrastructure"):
                raise ValueError(f"Unjustified retry after terminal result for {rid}")
        history[rid] = [str(attempts[index]["status"]) for index in ordered]
        record = attempts[ordered[-1]] if ordered else None
        status = str(record["status"]) if record is not None else "incomplete"
        counts[status] += 1
        if (record is None or
            (status in {"timeout", "incomplete"} and int(record.get("attempt", 0)) < retry_limit and
             record.get("failure_kind", "infrastructure") == "infrastructure") or
            (status == "crash" and record.get("failure_kind") == "infrastructure" and
             int(record.get("attempt", 0)) < retry_limit)):
            remaining.append(rid)
    assert sum(counts.values()) == len(planned)
    return {"manifest_sha": manifest["manifest_sha"], "planned": len(planned),
            "status_counts": counts, "remaining": remaining,
            "invalid_records": invalid, "attempt_history": history,
            "retry_count": sum(max(attempts) for attempts in seen.values())}


def restore_shard_records(manifest: Mapping[str, object], shard_id: str,
                          source_root: Path, output_root: Path) -> int:
    """Recover verified prior attempt records without crossing shard boundaries.

    Every previous GitHub Actions artifact remains immutable. Only the latest
    justified attempt is restored into this VM's active checkpoint directory.
    """
    expected = {str(run["run_id"]) for run in manifest["runs"]
                if str(run["shard"]) == str(shard_id)}
    prior = sorted(Path(source_root).rglob("*.json")) if Path(source_root).exists() else []
    verified: list[tuple[Path, dict[str, object]]] = []
    for path in prior:
        if path.stem not in expected:
            continue
        record = read_record(path)
        if record["run_id"] != path.stem:
            raise ValueError(f"Artifact path/run identity mismatch: {path}")
        verified.append((path, record))
    resume_plan(manifest, [path for path, _ in verified])
    chosen: dict[str, dict[str, object]] = {}
    for _, record in verified:
        rid = str(record["run_id"])
        if rid not in chosen or int(record.get("attempt", 0)) > int(chosen[rid].get("attempt", 0)):
            chosen[rid] = record
    for rid, record in chosen.items():
        dest = Path(output_root) / f"{rid}.json"
        if dest.exists() and read_record(dest) != record:
            raise ValueError(f"Refusing to overwrite conflicting restored record: {rid}")
        if not dest.exists():
            atomic_record(dest, record)
    return len(chosen)


def execute_shard(manifest_path: Path, shard_id: str, output_root: Path) -> int:
    """Run S0 identities separately, atomically persisting each completion.

    The local VM directory is NOT assumed to survive a sudden VM failure;
    artifacts must be uploaded in an Actions ``if: always()`` step.
    """
    from .audit import evaluate_candidate_set
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("manifest_sha") != _manifest_sha(manifest):
        raise ValueError("Manifest checksum mismatch")
    soft_limit = min(int(manifest["soft_deadline_minutes"]) * 60,
                     int(os.environ.get("NIR1_SOFT_DEADLINE_SECONDS", 18000)))
    deadline = time.monotonic() + soft_limit
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for run in manifest["runs"]:
        if str(run["shard"]) != str(shard_id):
            continue
        path = output_root / f'{run["run_id"]}.json'
        previous_attempt = -1
        if path.exists():
            existing = read_record(path)
            if existing["run_id"] != run["run_id"]:
                raise ValueError("run_id collision in artifact path")
            previous_attempt = int(existing.get("attempt", 0))
            if existing["status"] in {"ok", "unsupported"} or (
                    existing.get("failure_kind") != "infrastructure") or (
                    previous_attempt >= int(manifest.get("max_infrastructure_retries", 2))):
                continue
        if time.monotonic() >= deadline:
            break  # explicit incomplete in resume_plan; upload already-written files
        start = time.monotonic()
        result: dict[str, object] = {"run_id": run["run_id"],
                                     "manifest_sha": manifest["manifest_sha"],
                                     "attempt": previous_attempt + 1}
        try:
            if manifest["stage"] == "S0":
                rows = evaluate_candidate_set(str(run["case"]),
                                              [str(run["method"])],
                                              int(run["data_seed"]))
                if any(not row["solver_converged"] for row in rows):
                    result.update(status="crash", failure_kind="algorithm",
                                  reason="Candidate solver did not certify KKT convergence", rows=rows)
                else:
                    result.update(status="ok", rows=rows)
            else:
                # PIC requires a FRESH interpreter for every EPDE evolution:
                # process-global caches, operator parameters and RNG state
                # must never bleed from the preceding candidate/seed. This is
                # part of scientific independence, not an optional slowdown.
                with tempfile.TemporaryDirectory(prefix=".nir1-child-", dir=output_root) as tmp:
                    raw_file = Path(tmp) / "full-search.json"
                    argv = [sys.executable, "-m", "projects.nir1_stability.nir1.cli",
                            "epde", "--dataset", str(run["case"]),
                            "--variant", str(run["method"]),
                            "--seed", str(run["optimizer_seed"]),
                            "--data-seed", str(run["data_seed"]),
                            "--noise", str(manifest.get("noise", 0.)),
                            "--overrides-json", json.dumps(manifest.get("search_overrides", {})),
                            "--output", str(raw_file)]
                    timeout = max(1., min(deadline - time.monotonic(),
                                          float(manifest.get("max_run_minutes", 120)) * 60))
                    try:
                        process = subprocess.run(argv, capture_output=True, text=True,
                                                 timeout=timeout, check=False)
                    except subprocess.TimeoutExpired:
                        result.update(status="timeout", failure_kind="infrastructure",
                                      reason=f"Fresh EPDE subprocess exceeded {timeout:.1f}s")
                        result["elapsed_seconds"] = time.monotonic() - start
                        atomic_record(path, result)
                        continue
                    if not raw_file.exists():
                        result.update(status="crash", failure_kind="infrastructure",
                                      reason="No atomic EPDE record returned by child",
                                      returncode=process.returncode,
                                      stderr=process.stderr[-2000:])
                        result["elapsed_seconds"] = time.monotonic() - start
                        atomic_record(path, result)
                        continue
                    record = json.loads(raw_file.read_text(encoding="utf-8"))
                    result["child_returncode"] = process.returncode
                status = str(record["status"])
                if status == "ok":
                    result.update(status="ok", rows=[record])
                elif status == "unsupported":
                    result.update(status="unsupported", failure_kind="data_support",
                                  reason=record.get("reason", "Unsupported PIC system"),
                                  rows=[record])
                else:
                    result.update(status="crash", failure_kind="algorithm",
                                  reason=record.get("error", "PIC runner error"),
                                  rows=[record])
        except Exception as error:
            result.update(status="crash", failure_kind="algorithm",
                          error_type=type(error).__name__, reason=str(error)[:400],
                          traceback=traceback.format_exc(limit=2)[-1300:])
        result["elapsed_seconds"] = time.monotonic() - start
        atomic_record(path, result)
    return 0
