"""DeepMoD and DISCOVER from the same pinned PDEComp revision as PySINDy.

The native protocol is the default; fixed-library runs are also available.
Framework sources are downloaded
on demand and verified against pdecomp_sources.json, without installing their
old, mutually incompatible dependency manifests into the Streamlit process.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import re
import secrets
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

from . import pdecomp
from .baselines import UnsupportedProblem

METHODS = ('deepmod', 'discover')
LABELS = {'deepmod': 'DeepMoD', 'discover': 'DISCOVER'}
MANIFEST = Path(__file__).with_name('pdecomp_sources.json')
BRIDGE = Path(__file__).with_name('_pdecomp_methods_bridge.py')
# DISCOVER's upstream wrapper accepts scalar equations only.
DISCOVER_UNSUPPORTED = {'pdecomp_lv', 'pdecomp_lorenz', 'pdecomp_ns'}


def unsupported_reason(method, kind, variables, protocol='native', orders=None):
    if method not in METHODS:
        raise ValueError('Unknown PDEComp method')
    if method == 'discover' and (variables != 1 or kind not in ('ode', 'pde_1d')):
        return 'DISCOVER supports scalar ODEs and PDEs in one space dimension'
    if method == 'deepmod' and protocol == 'native':
        if variables != 1 or kind != 'pde_1d':
            return "DeepMoD native uses stock Library1D and requires a scalar 1D PDE"
        if orders and any(k != 1 for k in orders.values()):
            return 'DeepMoD native requires a first time derivative target (u_t)'
    if method == 'deepmod' and kind not in ('ode', 'ode_system', 'pde_1d', 'pde_2d'):
        return 'DeepMoD supports ODEs and PDEs in up to two space dimensions'
    return None


def source_dir(method):
    root = pdecomp.source_dir()
    groups = json.loads(MANIFEST.read_text(encoding='utf-8'))[method]

    def obtain(item):
        repo, revision, name, destination, digest = item
        path = root / destination
        if path.is_file() and pdecomp._sha256(path) == digest:
            return
        if os.environ.get('PDECOMP_DIR'):
            raise pdecomp.PdecompUnavailable(f'Missing or modified {destination} in PDECOMP_DIR')
        url = f'https://raw.githubusercontent.com/{repo}/{revision}/{name}'
        with urllib.request.urlopen(url, timeout=30) as response:
            payload = response.read(2_000_000)
        if hashlib.sha256(payload).hexdigest() != digest:
            raise pdecomp.PdecompUnavailable(f'Checksum mismatch: {destination}')
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + '.' + secrets.token_hex(6) + '.part')
        try:
            temp.write_bytes(payload)
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(obtain, groups))
    return root


def validate_overrides(method, overrides, protocol='native'):
    """Only numerical search controls; no executable paths or framework switches."""
    if method not in METHODS:
        raise ValueError('Unknown PDEComp method')
    if protocol not in pdecomp.PROTOCOLS:
        raise ValueError('Unknown protocol')
    if overrides is None:
        return {}
    if not isinstance(overrides, dict):
        raise ValueError('Method settings must be a mapping')
    limits = ({'threshold': (0, 1e4), 'pdefind_lam': (0, 1e4),
               'pdefind_dtol': (1e-12, 1e4), 'coefficient_tol': (0, 1e4)}
              if method == 'deepmod' else
              {'n_samples': (10, 50000), 'batch_size': (10, 500),
               'epsilon': (0.001, 1), 'max_length': (1, 64), 'max_add_count': (1, 15)})
    if method == 'deepmod' and protocol == 'native':
        limits = {'max_iterations': (1, 10000), 'write_iterations': (1, 1000),
                  'learning_rate': (1e-6, 0.1), 'max_samples': (100, 10000),
                  'poly_order': (1, 4), 'diff_order': (1, 4),
                  'threshold': (0, 1), 'coefficient_tol': (0, 1e4), 'train_fraction': (0.1, 0.9)}
    allowed = set(limits) | ({'type'} if method == 'deepmod' and protocol == 'fixed' else set())
    if set(overrides) - allowed:
        raise ValueError('Unknown method setting')
    clean = {}
    for key, value in overrides.items():
        if key == 'type':
            if value not in ('pdefind', 'threshold'):
                raise ValueError('DeepMoD estimator must be pdefind or threshold')
        else:
            low, high = limits[key]
            if type(value) not in (int, float) or not low <= value <= high:
                raise ValueError(f'{key} must be between {low} and {high}')
            if key in ('n_samples', 'batch_size', 'max_length', 'max_add_count', 'max_iterations',
                       'write_iterations', 'max_samples', 'poly_order', 'diff_order'):
                if int(value) != value:
                    raise ValueError(f'{key} must be an integer')
                value = int(value)
        clean[key] = value
    return clean


def _bridge(request, timeout=600):
    root = source_dir(request['method'])
    env = dict(os.environ, PYTHONIOENCODING='utf-8', MPLBACKEND='Agg',
               TF_CPP_MIN_LOG_LEVEL='2', TF_USE_LEGACY_KERAS='1',
               TF_NUM_INTRAOP_THREADS='1', TF_NUM_INTEROP_THREADS='1',
               NUMBA_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    env.pop('PYTHONPATH', None)
    request = dict(request, root=str(root))
    with tempfile.TemporaryDirectory() as tmp:
        request['output'] = str(Path(tmp) / 'result.json')
        proc = subprocess.Popen([sys.executable, str(BRIDGE), json.dumps(request)],
                                cwd=tmp, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, encoding='utf-8', errors='replace')
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            stop_process_tree(proc)
            proc.communicate(timeout=10)
            raise TimeoutError(f"{LABELS[request['method']]} exceeded the {timeout:g} s time limit. "
                               'Use Quick demo or reduce the training budget.') from None
        if proc.returncode:
            tail = (stderr or stdout).strip().splitlines()[-8:]
            raise RuntimeError(f"PDEComp {LABELS[request['method']]} failed: " + ' | '.join(tail))
        return json.loads(Path(request['output']).read_text(encoding='utf-8'))


def stop_process_tree(proc):
    """Windows venv launchers can have a second Python process below them."""
    import psutil
    try:
        children = psutil.Process(proc.pid).children(recursive=True)
        for child in reversed(children):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        proc.kill()
        psutil.wait_procs(children, timeout=5)
    except psutil.NoSuchProcess:
        pass


def describe(method, dataset=None, overrides=None, protocol='native', *, kind='ode', orders=None, seed=0):
    names = list(orders or {'u': 1})
    targets = [{'name': f"{'u' if len(names) == 1 else 'x' + str(i)}_{'t' * (orders or {}).get(v, 1)}",
                'variable': 'u' if len(names) == 1 else f'x{i}', 'axis': 't',
                'order': (orders or {}).get(v, 1)} for i, v in enumerate(names)]
    return _bridge({'method': method, 'filename': pdecomp.PDECOMP_FILES.get(dataset),
                    'protocol': protocol, 'describe': True,
                    'seed': int(seed),
                    'kind': kind, 'targets': targets,
                    'overrides': validate_overrides(method, overrides, protocol)}, 120)['settings']


def run(problem, data, method, seed=0, orders=None, overrides=None, protocol='native'):
    targets = dict(pdecomp.target_orders(problem), **(orders or {}))
    reason = unsupported_reason(method, problem.kind, len(problem.variables), protocol, targets)
    if reason:
        raise UnsupportedProblem(problem.name, reason)
    # PDEComp's Navier-Stokes record has a specialised supported library.
    if problem.kind == 'pde_2d' and not pdecomp.supported(problem.name):
        raise UnsupportedProblem(problem.name, 'DeepMoD 2D support requires PDEComp Navier-Stokes data')
    fields, x, y, z, t = pdecomp._layout(problem, data)
    if method == 'discover' or protocol == 'fixed':
        # These pinned runners use a single coordinate step for each axis.
        # Autograd in native DeepMoD instead receives the actual coordinates.
        for axis, coordinates in (('t', t), ('x', x), ('y', y), ('z', z)):
            if coordinates is None:
                continue
            grid = np.asarray(coordinates)
            values = np.asarray(grid, dtype=float)
            steps = np.diff(values)
            precision = np.finfo(grid.dtype).eps if grid.dtype.kind == 'f' else np.finfo(float).eps
            # Upload validation promotes coordinates to float64 without undoing
            # float32 rounding. Recover that precision only for an exact round trip.
            if (grid.dtype.kind == 'f' and precision < np.finfo(np.float32).eps
                    and np.all(np.abs(values) <= np.finfo(np.float32).max)
                    and np.array_equal(values, values.astype(np.float32).astype(float))):
                precision = np.finfo(np.float32).eps
            # Float32 linspace steps vary slightly after rounding. A large
            # coordinate origin must not make material spacing changes pass.
            tolerance = (min(4 * precision * float(np.max(np.abs(values))), abs(float(steps[0])) * 1e-3)
                         if steps.size else 0.0)
            if (not steps.size or not np.isfinite(steps).all() or not (steps > 0).all()
                    or not np.allclose(steps, steps[0], rtol=1e-5, atol=tolerance)):
                raise UnsupportedProblem(problem.name, f'{LABELS[method]} {protocol} requires a uniform {axis} grid '
                                         'for finite differences; resample the data to evenly spaced coordinates')
    if set(targets) != set(problem.variables) or any(type(v) is not int or v not in (1, 2) for v in targets.values()):
        raise ValueError('Target orders must be 1 or 2 for each variable')
    if len(targets) > 1 and any(v != 1 for v in targets.values()):
        raise UnsupportedProblem(problem.name, 'Systems require first time derivatives')
    names = ['u'] if len(targets) == 1 else [f'x{i}' for i in range(len(targets))]
    custom_targets = [{'name': f"{n}_{'t' * targets[v]}", 'variable': n, 'axis': 't', 'order': targets[v]}
                      for n, v in zip(names, problem.variables)]
    with tempfile.TemporaryDirectory() as tmp:
        arrays = {'t': t}
        arrays.update({k: v for k, v in (('x', x), ('y', y), ('z', z)) if v is not None})
        if isinstance(fields, list):
            arrays.update({f'field_{i}': f for i, f in enumerate(fields)})
        else:
            arrays['field'] = fields
        path = Path(tmp) / 'input.npz'
        np.savez(path, **arrays)
        reply = _bridge({'method': method, 'filename': pdecomp.PDECOMP_FILES.get(problem.name),
                         'protocol': protocol,
                         'input': str(path), 'seed': int(seed), 'targets': custom_targets,
                         'kind': problem.kind, 'overrides': validate_overrides(method, overrides, protocol)},
                        timeout=None if os.environ.get('EPDE_LOCAL_FULL_RUN') == '1' else 600)
    raw = reply['result']
    if method == 'discover' and protocol == 'native' and problem.kind == 'ode':
        # PDEComp reuses DISCOVER's spatial grammar for ODEs, where x1 is
        # time. Its lookup table misses products such as u u_x and u_x^2.
        # Fix canonical derivative labels without changing the original model
        # expression or coefficients; retain upstream feature names as well.
        features = [[re.sub(r'\bu_(x{1,4})\b', lambda m: 'u_' + 't' * len(m[1]), f)
                     for f in fs] for fs in raw['features']]
        if features != raw['features']:
            raw = dict(raw, features=features, original_features=raw['features'])
    # Native DISCOVER may discover functions outside the EPDE token grammar.
    # Keep those equations and its original reward, without claiming that our
    # structural metrics or numerical solver can interpret them.
    scoring_note = None
    try:
        texts = [pdecomp.to_epde_texts(problem, raw)]
    except (KeyError, ValueError):
        if method != 'discover' or protocol != 'native':
            raise
        texts = [[(' + '.join(f'{float(c)!r} * {f}' for c, f in zip(cs, fs) if abs(c) > 1e-12)
                   or '0.0') + ' = ' + target
                  for target, cs, fs in zip(raw['targets'], raw['coefficients'], raw['features'])]]
        scoring_note = 'Native DISCOVER found an expression outside the EPDE token grammar; structural scoring and solution reconstruction are unavailable.'
    info = {'protocol': protocol, 'source': f'{pdecomp.PDECOMP_REPO}@{pdecomp.PDECOMP_REV}',
            'pdecomp_file': pdecomp.PDECOMP_FILES.get(problem.name),
            'settings': reply['settings'], 'overrides': overrides or {},
            'library_size': (dict(zip(raw['targets'], pdecomp.library_size(raw)))
                             if protocol == 'fixed' or method == 'deepmod' else None),
            'result': raw,
            'differentiation': 'autograd' if method == 'deepmod' and protocol == 'native' else 'finite_difference',
            # The fitted neural network is not retained by the bridge. The
            # app evaluates equation balance independently on the observations.
            'reconstruction_differentiation': 'finite_difference', 'scoring_note': scoring_note}
    return texts, [None], reply['fit_seconds'], info


def score(texts, objectives, truths, baseline):
    from . import metrics
    if baseline.get('scoring_note'):
        return {'selected_index': 0, 'scoring_note': baseline['scoring_note']}
    return metrics.score_run(texts, objectives, truths)
