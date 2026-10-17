"""User-consented loopback worker for the *existing* EPDE and PySINDy libraries.

Run from the repository's Python environment, never from a browser:
    python projects/pic/app/local_worker_service.py

Browser requests are authenticated and restricted to the known demo origin.
Every job executes in an independent, time-bounded child process and uses
the ordinary production scientific adapters, not an approximation.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import secrets
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

APP = Path(__file__).resolve().parent
PIC = APP.parent
REPO = PIC.parent.parent
sys.path[:0] = [str(APP), str(PIC), str(REPO)]

HOST = "127.0.0.1"
PORT = 8765
MAX_REQUEST = 6 * 1024 * 1024
MAX_RESPONSE = 2 * 1024 * 1024
TIMEOUT = 600
VALID_ORIGINS = {
    "https://epde-explorer-demo.streamlit.app",
    "http://localhost:8501",
    "http://127.0.0.1:8501",
}
BUSY = threading.Lock()


def _execute(request, folder):
    """Child process. The parent validates the outer request before starting it."""
    from epde_bench.datasets import REGISTRY
    from support.discovery_lab import contract, worker
    from support.discovery_lab.lab_settings import validate_public_search
    from epde_bench.config import load_config, resolve_for_problem
    from epde_bench.datasets import load

    if request.get("method") not in ("epde", "pdecomp"):
        raise ValueError("Local worker supports EPDE and PySINDy only")
    if request.get("source") == "record":
        dataset = request.get("dataset")
        if not isinstance(dataset, str) or dataset not in REGISTRY:
            raise ValueError("Unknown dataset")
        cfg = load_config(dataset, request.get("variant", "default"),
                          overrides=request.get("overrides") or {})
        problem = load(dataset, **cfg.get("loader", {}))
        validate_public_search(resolve_for_problem(cfg, problem), problem.shape)
        runreq = dict(request)
        record = worker._run_record(runreq)
    elif request.get("source") == "upload":
        import yaml
        from support.custom_run import validate_arrays
        import numpy as np
        from io import BytesIO
        payload = request.get("data_b64")
        settings = request.get("settings")
        if not isinstance(payload, str) or len(payload) > MAX_REQUEST * 2:
            raise ValueError("Input data invalid")
        if not isinstance(settings, dict) or not isinstance(settings.get("search"), dict):
            raise ValueError("Missing numerical settings")
        data = base64.b64decode(payload, validate=True)
        if len(data) > 3 * 1024 * 1024:
            raise ValueError("Input data exceeds local transfer limit")
        with np.load(BytesIO(data), allow_pickle=False) as z:
            if len(z.files) > 15:
                raise ValueError("Too many arrays")
            arrays = {k: z[k] for k in z.files}
        validate_arrays(arrays)
        if sum(v.size for v in arrays.values()) > 300_000:
            raise ValueError("Input data exceeds 300,000 numbers")
        owner = folder / "owner"
        target = owner / "jobs" / "job"
        target.mkdir(parents=True)
        files = owner / "uploads" / "local"
        files.mkdir(parents=True)
        np.savez_compressed(files / "data.npz", **arrays)
        (files / "settings.yaml").write_text(yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")
        runreq = {
            "method": request["method"],
            "source": "upload",
            "dataset": "custom",
            "data_path": str(files / "data.npz"),
            "config_path": str(files / "settings.yaml"),
            "noise": float(request.get("noise", 0)),
            "seed": int(request.get("seed", 0)),
        }
        record = worker._run_upload(runreq, target)
    else:
        raise ValueError("Invalid source")

    record = contract.normalize(worker._sanitize(worker._finite_objectives(record)),
                                request["method"])
    if record.get("status") == "ok":
        try:
            record["evaluation"] = worker._sanitize(worker.evaluate(record))
        except Exception:
            record["evaluation"] = {"error": "Evaluation unavailable"}
    # Do not publish the local user's file system structure to Streamlit Cloud.
    if record.get("dataset") == "custom":
        record["data_file"] = "input_data.npz"
    record["execution_location"] = "user_computer"
    record["worker_implementation"] = "native-python"
    return record


def _child_main(source: Path, output: Path) -> int:
    try:
        request = json.loads(source.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory(prefix="epde_local_job_") as temp:
            record = _execute(request, Path(temp))
        output.write_text(json.dumps({"record": record}, allow_nan=False),
                          encoding="utf-8")
        return 0
    except Exception as exc:
        output.write_text(json.dumps({"error": f"Local worker: {type(exc).__name__}"}),
                          encoding="utf-8")
        return 1


class Handler(BaseHTTPRequestHandler):
    server_version = "EPDELocal/1.0"

    def log_message(self, format, *args):
        # Suppress access log details (the content may involve private data).
        return

    def _origin(self):
        return self.headers.get("Origin", "")

    def _headers(self, status, *, cors=True):
        self.send_response(status)
        if cors and self._origin() in VALID_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", self._origin())
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, X-EPDE-Token")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Private-Network", "true")
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()

    def _reply(self, status, data):
        raw = json.dumps(data, allow_nan=False).encode("utf-8")
        self._headers(status)
        self.wfile.write(raw)

    def _authorized(self):
        return (self._origin() in VALID_ORIGINS
                and secrets.compare_digest(self.headers.get("X-EPDE-Token", ""),
                                           self.server.auth_token))

    def do_OPTIONS(self):
        if self._origin() not in VALID_ORIGINS:
            self._headers(403, cors=False)
            return
        self._headers(204)

    def do_GET(self):
        if self.path != "/health" or not self._authorized():
            return self._reply(403, {"error": "Forbidden"})
        return self._reply(200, {"ok": True, "methods": ["epde", "pdecomp"],
                                 "execution_location": "user_computer",
                                 "protocol": "discovery-local/1.0"})

    def do_POST(self):
        if self.path != "/run" or not self._authorized():
            return self._reply(403, {"error": "Forbidden"})
        if not BUSY.acquire(blocking=False):
            return self._reply(429, {"error": "One local search at a time"})
        try:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            if not 0 < length <= MAX_REQUEST:
                return self._reply(413, {"error": "Request must be 1–6 MiB"})
            try:
                request = json.loads(self.rfile.read(length))
            except (ValueError, UnicodeError):
                return self._reply(400, {"error": "Invalid JSON"})
            if not isinstance(request, dict) or request.get("method") not in ("epde", "pdecomp"):
                return self._reply(400, {"error": "Unsupported local method"})
            if request.get("source") not in ("record", "upload"):
                return self._reply(400, {"error": "Invalid data source"})
            with tempfile.TemporaryDirectory(prefix="epde_local_request_") as temp:
                inp = Path(temp) / "request.json"
                out = Path(temp) / "result.json"
                inp.write_text(json.dumps(request), encoding="utf-8")
                environment = dict(os.environ)
                environment["OPENBLAS_NUM_THREADS"] = "1"
                environment["OMP_NUM_THREADS"] = "1"
                environment["MKL_NUM_THREADS"] = "1"
                try:
                    proc = subprocess.run(
                        [sys.executable, str(Path(__file__)), "--execute", str(inp), str(out)],
                        cwd=str(REPO), env=environment, capture_output=True,
                        timeout=TIMEOUT, check=False)
                except subprocess.TimeoutExpired:
                    return self._reply(408, {"error": "Local search exceeded 10 minutes"})
                if not out.exists() or out.stat().st_size > MAX_RESPONSE:
                    return self._reply(500, {"error": "No valid result; see worker terminal"})
                answer = json.loads(out.read_text(encoding="utf-8"))
                if "error" in answer:
                    return self._reply(422, answer)
                return self._reply(200, answer)
        except (OSError, ValueError) as exc:
            return self._reply(500, {"error": f"{type(exc).__name__}: local request failed"})
        finally:
            BUSY.release()


def main():
    parser = argparse.ArgumentParser(description="Local EPDE/PySINDy worker on the user's PC")
    parser.add_argument("--execute", nargs=2, metavar=("REQUEST", "OUTPUT"))
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()
    if args.execute:
        return _child_main(Path(args.execute[0]), Path(args.execute[1]))
    if args.port != PORT:
        parser.error("Only the documented local port 8765 is supported for the demo")
    token = secrets.token_urlsafe(24)
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    server.auth_token = token
    print("EPDE Local Worker — native Python; your PC performs the search.")
    print("Address: http://127.0.0.1:8765")
    print("Local access code (enter in Lab; do not share):")
    print(token, flush=True)
    print("Close this terminal or press Ctrl+C to stop access.")
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
