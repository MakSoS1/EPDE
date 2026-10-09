"""Isolate the short lesson's EPDE globals from every Streamlit session."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


APP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_DIR.parents[2]


def run_optimizer_subprocess():
    """Return the existing demo's result from a fresh process; clean up all IPC files."""
    timeout = 60
    with tempfile.TemporaryDirectory(prefix='epde-optimizer-') as temporary:
        directory = Path(temporary)
        result_path = directory / 'result.json'
        log_path = directory / 'execution.log'
        environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                           MKL_NUM_THREADS='1', MPLBACKEND='Agg',
                           MPLCONFIGDIR=str(directory / 'matplotlib'), PYTHONIOENCODING='utf-8')
        try:
            with log_path.open('w', encoding='utf-8') as log:
                completed = subprocess.run(
                    [sys.executable, str(APP_DIR / 'support' / 'optimizer_run.py'), str(result_path)],
                    cwd=REPO_ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                    timeout=timeout, check=False)
        except subprocess.TimeoutExpired as exc:
            # subprocess.run kills and waits for its child before propagating timeout.
            raise RuntimeError(f'The short evolution exceeded {timeout} seconds.') from exc
        if completed.returncode != 0:
            detail = log_path.read_text(encoding='utf-8', errors='replace')[-6000:].strip()
            raise RuntimeError(f'The optimizer child process failed: {detail or completed.returncode}')
        try:
            result = json.loads(result_path.read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            raise RuntimeError('The optimizer child process did not save a readable result.') from exc
        if not isinstance(result, dict) or not all(key in result for key in
                                                   ('history', 'rows', 'equations', 'seconds')):
            raise RuntimeError('The optimizer child process saved an incomplete result.')
        return result
