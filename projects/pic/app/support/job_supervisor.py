"""Detached worker owner: record exit status even when the Streamlit process exits."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil


def save(path, record):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(record, indent=1), encoding='utf-8')
    temporary.replace(path)


def supervise(path):
    path = Path(path)
    job = json.loads(path.read_text(encoding='utf-8'))
    job.update(pid=os.getpid(), process_created=psutil.Process().create_time(), state='running')
    save(path, job)
    try:
        child = subprocess.Popen(job['cmd'], start_new_session=os.name != 'nt')
        job['worker_pid'] = child.pid
        job['worker_created'] = psutil.Process(child.pid).create_time()
        save(path, job)
        code = child.wait()
        stopped = path.with_suffix('.stop').exists()
        job.update(exit_code=code, ended=time.time(), state='stopped' if stopped else ('finished' if code == 0 else 'error'))
    except Exception as exc:
        job.update(exit_code=None, ended=time.time(), state='error', error=repr(exc))
    save(path, job)


if __name__ == '__main__':
    supervise(sys.argv[1])
