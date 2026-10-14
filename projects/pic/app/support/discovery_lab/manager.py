"""Long-running single-host process supervisor. Run only one per deployment."""
from __future__ import annotations
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# Standalone script; import its package from the application root.
APP = Path(__file__).resolve().parents[2]
PIC = APP.parent
REPO = PIC.parent.parent
sys.path[:0] = [str(APP), str(PIC), str(REPO)]
from support.discovery_lab import queue

MAX_WORKERS = max(1, min(2, int(os.environ.get("EPDE_DEMO_WORKERS", "2"))))
TIMEOUT = max(30, min(600, int(os.environ.get("EPDE_DEMO_TIMEOUT_SECONDS", "180"))))


def _terminate(proc):
    if proc.poll() is not None:
        return
    try:
        if os.name != "nt":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            import psutil
            parent = psutil.Process(proc.pid)
            for child in parent.children(recursive=True):
                child.kill()
            parent.kill()
    except (OSError, ProcessLookupError):
        pass


def main():
    lock = queue._read(queue.MANAGER_LOCK)
    if lock and queue._pid_alive(lock.get("pid")) and lock["pid"] != os.getpid():
        return 1
    queue._write(queue.MANAGER_LOCK, {"pid": os.getpid(), "started": time.time()})
    running = {}
    try:
        idle_since = time.time()
        while True:
            pending = []
            for path in queue._records():
                job = queue._read(path)
                if not job:
                    continue
                folder = path.parent
                if job.get("state") == "queued":
                    if (folder / "cancel").exists():
                        job.update(state="stopped", ended=time.time())
                        queue._write(path, job)
                    else:
                        pending.append((job.get("created", 0), path, job))
            for path, (proc, start) in list(running.items()):
                job = queue._read(path) or {}
                canceled = (path.parent / "cancel").exists()
                timed_out = time.time() - start > TIMEOUT
                if canceled or timed_out:
                    _terminate(proc)
                exitcode = proc.poll()
                if exitcode is not None:
                    job.update(state=("stopped" if canceled else "timeout" if timed_out
                                      else "finished" if exitcode == 0 else "error"),
                               exit_code=exitcode, ended=time.time())
                    # Worker writes normalized result; manager's state describes process.
                    queue._write(path, job)
                    del running[path]
            for _, path, job in sorted(pending)[:max(0, MAX_WORKERS - len(running))]:
                folder = path.parent
                with open(folder / "worker.log", "ab") as log:
                    proc = subprocess.Popen([sys.executable,
                                             str(Path(__file__).with_name("worker.py")), str(folder)],
                                            cwd=str(REPO), stdout=log, stderr=subprocess.STDOUT,
                                            start_new_session=os.name != "nt",
                                            creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0))
                job.update(state="running", started=time.time(), pid=proc.pid)
                queue._write(path, job)
                running[path] = (proc, time.time())
            if not running and not pending:
                if time.time() - idle_since > 60:
                    return 0
            else:
                idle_since = time.time()
            time.sleep(.5)
    finally:
        for proc, _ in running.values():
            _terminate(proc)
        lock = queue._read(queue.MANAGER_LOCK)
        if lock and lock.get("pid") == os.getpid():
            queue.MANAGER_LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
