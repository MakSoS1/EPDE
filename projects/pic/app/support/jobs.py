"""Background jobs started from the app: one search, a search on uploaded data,
a campaign. Each job is a separate process with a log; its description lives in
``results/app/jobs/<id>.json`` so jobs survive a page reload or a restart of the app.
"""

import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

from epde_bench.paths import PIC_DIR, REPO_ROOT, RESULTS_DIR

APP_DIR = PIC_DIR / 'app'
APP_RESULTS = RESULTS_DIR / 'app'

JOBS_DIR = APP_RESULTS / 'jobs'
RUNS_DIR = APP_RESULTS / 'runs'
_PROCESSES = {}


def _flatten(overrides, prefix=''):
    out = []
    for key, value in (overrides or {}).items():
        path = f'{prefix}{key}'
        if isinstance(value, dict):
            out += _flatten(value, path + '.')
        else:
            out.append(f'{path}={json.dumps(value)}')
    return out


def _start(kind, title, cmd, out=None, extra=None):
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    job_id = datetime.now().strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:4]
    log = JOBS_DIR / f'{job_id}.log'
    env = dict(os.environ, PYTHONIOENCODING='utf-8', MPLBACKEND='Agg', OPENBLAS_NUM_THREADS='1')
    flags = 0
    if os.name == 'nt':
        flags = subprocess.CREATE_NEW_PROCESS_GROUP
    job = {'id': job_id, 'kind': kind, 'title': title, 'cmd': cmd,
           'out': str(out) if out else None, 'log': str(log), 'started': time.time(), 'state': 'starting'}
    job.update(extra or {})
    path = JOBS_DIR / f'{job_id}.json'
    path.write_text(json.dumps(job, indent=1), encoding='utf-8')
    with open(log, 'w', encoding='utf-8') as fh:
        proc = subprocess.Popen([sys.executable, str(APP_DIR / 'support' / 'job_supervisor.py'), str(path)],
                                stdout=fh, stderr=subprocess.STDOUT, cwd=str(REPO_ROOT), env=env,
                                creationflags=flags, start_new_session=os.name != 'nt')
    _PROCESSES[job_id] = proc
    import psutil
    job.update(pid=proc.pid, process_created=psutil.Process(proc.pid).create_time())
    return job


def start_run(dataset, variant='default', noise=0.0, seed=0, overrides=None):
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = RUNS_DIR / f'{dataset}__{variant}__noise{noise:g}__seed{seed}__{uuid.uuid4().hex[:6]}.json'
    cmd = [sys.executable, str(PIC_DIR / 'bench.py'), 'run', dataset, '--variant', variant,
           '--noise', str(noise), '--seed', str(seed), '--out', str(out)]
    for item in _flatten(overrides):
        cmd += ['--set', item]
    from .common import dataset_label, variant_label
    title = f'{dataset_label(dataset)} · {variant_label(variant)} · noise {noise:g} % · seed {seed}'
    return _start('run', title, cmd, out, {'dataset': dataset, 'overrides': overrides or {}})


def start_custom(npz_path, config_path, title):
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = RUNS_DIR / f'custom__{Path(npz_path).stem}__{uuid.uuid4().hex[:6]}.json'
    cmd = [sys.executable, str(APP_DIR / 'support' / 'custom_run.py'), '--data', str(npz_path),
           '--config', str(config_path), '--out', str(out)]
    return _start('custom', title, cmd, out, {'dataset': 'custom'})


def start_campaign(name, datasets, variants, noise, seeds, workers):
    cmd = [sys.executable, str(PIC_DIR / 'bench.py'), 'campaign', '--datasets', ','.join(datasets),
           '--variants', ','.join(variants), '--noise', ','.join(f'{n:g}' for n in noise),
           '--seeds', seeds, '--workers', str(workers), '--name', name]
    return _start('campaign', f'campaign {name}', cmd, RESULTS_DIR / name, {'campaign': name})


def _alive(pid, created=None):
    import psutil
    if pid is None or created is None:
        return False  # old records have no safe identity; never trust a reused PID
    try:
        process = psutil.Process(pid)
        return abs(process.create_time() - created) < .001 and process.status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def _current(job):
    for key, proc in list(_PROCESSES.items()):
        if proc.poll() is not None:
            del _PROCESSES[key]
    if 'id' in job:
        try:
            stored = json.loads((JOBS_DIR / f"{job['id']}.json").read_text(encoding='utf-8'))
            return dict(job, **stored)
        except (OSError, ValueError):
            pass
    return job


def campaign_records(folder):
    """Only valid terminal records belonging to the planned grid count as done."""
    from epde_bench.campaign import run_stem
    try:
        planned = json.loads((Path(folder) / 'campaign.json').read_text(encoding='utf-8'))['jobs']
    except (OSError, ValueError, KeyError):
        return [], 0
    records = []
    for plan in planned:
        path = Path(folder) / 'runs' / plan['dataset'] / (run_stem(plan['variant'], plan['noise'], plan['seed']) + '.json')
        try:
            record = json.loads(path.read_text(encoding='utf-8'))
            if record.get('status') not in ('ok', 'unsupported', 'error', 'timeout'):
                continue
            if plan.get('identity') is not None and record.get('identity') != plan['identity']:
                continue
            records.append(record)
        except (OSError, ValueError, AttributeError):
            continue
    return records, len(planned)


def status(job):
    """Report process termination and output coverage separately from manifest creation."""
    job = _current(job)
    state = job.get('state')
    # The initial manifest is persisted before the detached supervisor can write its PID.
    if state == 'starting' and 0 <= time.time() - job.get('started', 0) < 10:
        return 'running'
    if state not in ('finished', 'stopped', 'error') and _alive(job.get('pid'), job.get('process_created')):
        return 'running'
    if state in ('stopped', 'error'):
        return state
    if 'exit_code' in job and job['exit_code'] != 0:
        return 'error'
    if job['kind'] == 'campaign':
        records, planned = campaign_records(job['out'])
        if not planned or len(records) != planned:
            return 'partial' if records else ('error' if state == 'finished' else 'stopped')
        return 'error' if any(r['status'] in ('error', 'timeout') for r in records) else 'ok'
    try:
        record = json.loads(Path(job['out']).read_text(encoding='utf-8'))
        return record.get('status', 'error')
    except (OSError, ValueError, TypeError, AttributeError):
        return 'error' if state == 'finished' else 'stopped'


def jobs(kind=None):
    out = []
    for path in sorted(JOBS_DIR.glob('*.json'), reverse=True):
        try:
            job = json.loads(path.read_text(encoding='utf-8'))
        except ValueError:
            continue
        if kind is None or job['kind'] in ((kind,) if isinstance(kind, str) else kind):
            out.append(job)
    return sorted(out, key=lambda job: job.get('started', 0), reverse=True)


def log_tail(job, lines=40):
    try:
        text = Path(job['log']).read_text(encoding='utf-8', errors='replace')
    except OSError:
        return ''
    return '\n'.join(text.splitlines()[-lines:])


def stop(job):
    import psutil
    job = _current(job)
    if not _alive(job.get('pid'), job.get('process_created')):
        return False
    try:
        parent = psutil.Process(job['pid'])
        if abs(parent.create_time() - job['process_created']) >= .001:
            return False
        # The worker has its own POSIX process group; the supervisor stays alive
        # to reap it and persist the exit code. Verify the worker identity too.
        if not _alive(job.get('worker_pid'), job.get('worker_created')):
            return False
        worker = psutil.Process(job['worker_pid'])
        (JOBS_DIR / f"{job['id']}.stop").touch()
        if os.name != 'nt':
            import signal
            os.killpg(worker.pid, signal.SIGKILL)
        else:
            for child in reversed(worker.children(recursive=True)):
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            worker.kill()
        return True
    except (psutil.Error, OSError):
        return False


def resume_campaign(job, retry_errors=False):
    job = _current(job)
    if status(job) == 'running':
        raise ValueError('This campaign is already running.')
    cmd = [arg for arg in job['cmd'] if arg != '--retry-errors']
    if retry_errors:
        cmd.append('--retry-errors')
    return _start('campaign', job['title'], cmd, job['out'], {'campaign': job['campaign']})


def campaign_progress(name):
    records, planned = campaign_records(RESULTS_DIR / name)
    return len(records), planned
