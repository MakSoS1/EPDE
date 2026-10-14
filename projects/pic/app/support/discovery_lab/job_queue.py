"""Process-safe local demo queue with capability-scoped owners.

Designed for a single trusted host and bounded demo load, not distributed
multi-tenancy. The manager owns children; workers never import Streamlit.
"""
from __future__ import annotations
import json
import os
import re
import secrets
import subprocess
import sys
import time
from pathlib import Path

from epde_bench.paths import RESULTS_DIR

ROOT = RESULTS_DIR / "app" / "discovery_lab"
MANAGER_LOCK = ROOT / "manager.lock"
OWNER_RE = re.compile(r"^[0-9a-f]{48}$")
JOB_RE = re.compile(r"^[0-9a-f]{32}$")
MAX_OWNER_JOBS = 40
MAX_OWNER_PENDING = 6
MAX_TOTAL_QUEUED = 30
MAX_OWNER_UPLOAD_BYTES = 24 * 1024 * 1024
MAX_LLM_DAILY_JOBS = 50


def new_owner() -> str:
    return secrets.token_hex(24)


def owner_path(owner: str) -> Path:
    if not OWNER_RE.fullmatch(owner):
        raise ValueError("Invalid session capability")
    return ROOT / "owners" / owner


def job_path(owner: str, job_id: str) -> Path:
    if not JOB_RE.fullmatch(job_id):
        raise ValueError("Invalid job id")
    return owner_path(owner) / "jobs" / job_id


def _write(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + secrets.token_hex(6) + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def _read(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _records():
    if not ROOT.exists():
        return []
    return list(ROOT.glob("owners/*/jobs/*/job.json"))


def _manager_alive():
    lock = _read(MANAGER_LOCK)
    return bool(lock and _pid_alive(lock.get("pid")))


def list_jobs(owner: str):
    """Jobs of one session, newest first.

    A server restart kills the manager and its workers: a job left ``running``
    with neither alive is marked ``interrupted`` (its result, if already
    written, is kept), and ``queued`` jobs restart the manager."""
    folder = owner_path(owner) / "jobs"
    paths = list(folder.glob("*/job.json")) if folder.exists() else []
    rows, waiting = [], False
    alive = None
    for path in paths:
        row = _read(path)
        if not row:
            continue
        if row.get("state") == "running":
            alive = _manager_alive() if alive is None else alive
            if not alive and not _pid_alive(row.get("pid")):
                row.update(state="interrupted", ended=time.time())
                _write(path, row)
        waiting |= row.get("state") == "queued"
        rows.append(row)
    if waiting and not (_manager_alive() if alive is None else alive):
        ensure_manager()
    return sorted(rows, key=lambda r: r.get("created", 0), reverse=True)


def delete_job(owner: str, job_id: str) -> bool:
    """Remove a finished job of this session (running jobs are cancelled first)."""
    import shutil
    folder = job_path(owner, job_id)
    job = get_job(owner, job_id)
    if not job:
        return False
    if job.get("state") in ("queued", "running"):
        return False
    shutil.rmtree(folder, ignore_errors=True)
    return True


def get_job(owner: str, job_id: str):
    return _read(job_path(owner, job_id) / "job.json")


def check_capacity(owner: str, method: str, count: int = 1):
    """Raise unless ``count`` more searches fit the session and server limits."""
    own_jobs = list_jobs(owner)
    if sum(r.get("state") in ("queued", "running") for r in own_jobs) + count > MAX_OWNER_PENDING:
        raise ValueError(f"At most {MAX_OWNER_PENDING} waiting or running searches per session; "
                         "wait for some to finish")
    if len(own_jobs) + count > MAX_OWNER_JOBS:
        raise ValueError("Session reached its stored result limit; delete old results")
    all_jobs = [_read(p) for p in _records()]
    if sum((r or {}).get("state") == "queued" for r in all_jobs) + count > MAX_TOTAL_QUEUED:
        raise ValueError("Server queue is full; retry later")
    if method == "llm":
        now = time.time()
        if sum(r.get("method") == "llm" and now - r.get("created", 0) < 3600
               for r in own_jobs) >= 3:
            raise ValueError("LLM demo limit: 3 requests per session per hour")
        if sum((r or {}).get("method") == "llm" and now - (r or {}).get("created", 0) < 86400
               for r in all_jobs) >= MAX_LLM_DAILY_JOBS:
            raise ValueError("LLM daily server budget reached")


def submit(owner: str, request: dict):
    """Create an immutable input manifest and a queued job; never accept a shell command."""
    if request.get("method") not in ("epde", "pysindy", "llm"):
        raise ValueError("Unknown algorithm")
    if request.get("source") not in ("record", "upload"):
        raise ValueError("Unknown data source")
    if not isinstance(request.get("dataset"), str) or len(request["dataset"]) > 80:
        raise ValueError("Invalid dataset")
    check_capacity(owner, request["method"])
    job_id = secrets.token_hex(16)
    folder = job_path(owner, job_id)
    folder.mkdir(parents=True, exist_ok=False)
    manifest = dict(request)
    manifest["owner"] = owner
    manifest["job_id"] = job_id
    _write(folder / "request.json", manifest)
    display = request.get("display") if isinstance(request.get("display"), dict) else {}
    _write(folder / "job.json", {"id": job_id, "created": time.time(), "state": "queued",
                                "method": request["method"], "source": request["source"],
                                "dataset": request["dataset"],
                                "label": str(display.get("label", request["dataset"]))[:160],
                                "input_key": str(display.get("input_key", ""))[:200],
                                "noise": float(request.get("noise", 0) or 0),
                                "seed": int(request.get("seed", 0) or 0),
                                "effort": str(request.get("effort", "standard"))[:20]})
    ensure_manager()
    return job_id


def cancel(owner: str, job_id: str):
    folder = job_path(owner, job_id)
    job = get_job(owner, job_id)
    if not job or job["state"] not in ("queued", "running"):
        return False
    (folder / "cancel").touch(exist_ok=True)
    return True


def result(owner: str, job_id: str):
    return _read(job_path(owner, job_id) / "result.json")


def save_upload(owner: str, arrays: dict, settings: dict):
    import numpy as np
    import yaml
    uploads_root = owner_path(owner) / "uploads"
    stored_bytes = sum(p.stat().st_size for p in uploads_root.rglob("*") if p.is_file()) if uploads_root.exists() else 0
    if stored_bytes >= MAX_OWNER_UPLOAD_BYTES:
        raise ValueError("Session upload quota exceeded")
    folder = uploads_root / secrets.token_hex(16)
    folder.mkdir(parents=True, exist_ok=False)
    data = folder / "data.npz"
    conf = folder / "settings.yaml"
    np.savez_compressed(data, **arrays)
    conf.write_text(yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")
    if stored_bytes + data.stat().st_size + conf.stat().st_size > MAX_OWNER_UPLOAD_BYTES:
        import shutil
        shutil.rmtree(folder)
        raise ValueError("Session upload quota exceeded")
    return data, conf


def _pid_alive(pid):
    if not isinstance(pid, int) or pid < 2:
        return False
    try:
        import psutil
        p = psutil.Process(pid)
        return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    except Exception:
        return False


def ensure_manager():
    """The exclusive manager lock prevents duplicate schedulers on a single host."""
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = _read(MANAGER_LOCK)
    if lock and _pid_alive(lock.get("pid")):
        return
    # Atomic launcher lock. If the original manager exited, reclaim it.
    if MANAGER_LOCK.exists():
        try:
            if time.time() - MANAGER_LOCK.stat().st_mtime < 10:
                return
            MANAGER_LOCK.unlink()
        except OSError:
            return
    launcher = ROOT / "launch.lock"
    try:
        fd = os.open(launcher, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        try:
            if time.time() - launcher.stat().st_mtime > 30:
                launcher.unlink()
        except OSError:
            pass
        return
    try:
        os.close(fd)
        logfile = open(ROOT / "manager.log", "ab")
        try:
            # Manage first with no user-controlled arguments.
            subprocess.Popen([sys.executable, str(Path(__file__).with_name("manager.py"))],
                             cwd=str(Path(__file__).resolve().parents[5]),
                             stdout=logfile, stderr=subprocess.STDOUT,
                             start_new_session=os.name != "nt",
                             creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0))
        finally:
            logfile.close()
        # Keep the launcher mutex during process startup. Without this wait,
        # two simultaneous browser sessions could start two managers.
        for _ in range(30):
            if MANAGER_LOCK.exists():
                break
            time.sleep(0.1)
    finally:
        launcher.unlink(missing_ok=True)


def save_client_result(owner: str, request: dict, record: dict):
    """Store a *validated* completed local run for the existing Results/Compare.

    The browser is not authoritative about method, input identity or storage path.
    It cannot submit arbitrary server commands or create a running server-side job.
    """
    from .contract import normalize
    from epde_bench.datasets import REGISTRY
    if not isinstance(request, dict) or not isinstance(record, dict):
        raise ValueError("Invalid local result")
    method = request.get("method")
    if method not in ("epde", "pysindy"):
        raise ValueError("Invalid local method")
    if record.get("execution_location") != "user_computer":
        raise ValueError("Result was not marked as a local run")
    payload = dict(record)
    if len(json.dumps(payload, default=str)) > 1_800_000:
        raise ValueError("Result exceeds browser transport limit")
    payload = normalize(payload, method)
    if payload.get("dataset") != request.get("dataset") or payload["method"] != method:
        raise ValueError("Local worker returned a different dataset or method")
    if request.get("source") == "record":
        if request.get("dataset") not in REGISTRY:
            raise ValueError("Invalid built-in dataset")
    elif request.get("source") != "upload" or request.get("dataset") != "custom":
        raise ValueError("Invalid data source")
    # A local user can provide forged candidate values (a capability they
    # already have through data uploads). Results are *not* used as training
    # ground truth or automatically published benchmark leaderboards.
    if len(list_jobs(owner)) >= MAX_OWNER_JOBS:
        raise ValueError("The private session's stored-result limit was reached")
    job_id = secrets.token_hex(16)
    folder = job_path(owner, job_id)
    folder.mkdir(parents=True, exist_ok=False)
    created = time.time()
    display = request.get("display") if isinstance(request.get("display"), dict) else {}
    state = "finished" if payload.get("status") in ("ok", "unsupported") else "error"
    _write(folder / "request.json", {
        "method": method, "source": request["source"],
        "dataset": request["dataset"], "variant": request.get("variant"),
        "noise": request.get("noise", 0), "seed": request.get("seed", 0),
        "effort": request.get("effort", "standard"),
        "execution_location": "user_computer",
    })
    _write(folder / "result.json", payload)
    _write(folder / "job.json", {
        "id": job_id, "created": created, "started": created, "ended": created,
        "state": state, "method": method, "source": request["source"],
        "dataset": request["dataset"],
        "label": str(display.get("label", request["dataset"]))[:160],
        "input_key": str(display.get("input_key", ""))[:200],
        "noise": float(request.get("noise", 0)),
        "seed": int(request.get("seed", 0)),
        "effort": str(request.get("effort", "standard"))[:60],
        "execution_location": "user_computer",
    })
    return job_id
