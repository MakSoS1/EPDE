"""One EPDE search on data uploaded in the app; writes the same JSON record as ``bench.py run``.

    python projects/pic/app/support/custom_run.py --data upload.npz --config settings.yaml --out record.json

``upload.npz`` holds ``axis_<k>`` (1-D coordinates, time first), ``axis_names``
and one ``var_<name>`` array per variable on the grid. ``settings.yaml`` holds the
``search`` section of a configuration (as in ``configs/*.yaml``) and optionally
``noise`` and ``seed``.
"""

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

PIC_DIR = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PIC_DIR), str(PIC_DIR.parent.parent)]

from epde_bench import env  # noqa: E402

env.pin_blas_threads(1)


def problem_from_npz(path):
    import numpy as np
    from epde_bench.problem import Problem, mesh
    with np.load(path, allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    axes, names, data = validate_arrays(arrays)
    kinds = {1: 'ode' if len(data) == 1 else 'ode_system', 2: 'pde_1d', 3: 'pde_2d', 4: 'pde_3d'}
    return Problem(name='custom', title=f'Uploaded data ({Path(path).stem})', kind=kinds[len(axes)],
                   source='real', grids=mesh(*axes), data=data, axis_names=tuple(names))


def validate_arrays(arrays):
    """Validate uploads before saving them or starting a worker."""
    import re
    import numpy as np
    axis_keys = [k for k in arrays if k.startswith('axis_') and k != 'axis_names']
    if not 1 <= len(axis_keys) <= 4 or set(axis_keys) != {f'axis_{i}' for i in range(len(axis_keys))}:
        raise ValueError('Supply 1–4 consecutive coordinate arrays: axis_0, axis_1, … (time first).')
    axes = []
    for i in range(len(axis_keys)):
        axis = np.asarray(arrays[f'axis_{i}'], dtype=float)
        if axis.ndim != 1 or axis.size < 2 or not np.isfinite(axis).all() or not (np.diff(axis) > 0).all():
            raise ValueError(f'axis_{i}: coordinates must be finite, strictly increasing and contain at least two points.')
        axes.append(axis)
    names = np.asarray(arrays.get('axis_names', ['t', 'x', 'y', 'z'][:len(axes)]))
    if names.ndim != 1 or len(names) != len(axes) or len(set(map(str, names))) != len(names):
        raise ValueError('axis_names must contain one unique name per coordinate axis.')
    names = [str(n) for n in names]
    if any(not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', n) for n in names):
        raise ValueError('Axis names must start with a letter and contain only letters, digits and underscores.')
    data = {}
    shape = tuple(len(a) for a in axes)
    for key, values in arrays.items():
        if not key.startswith('var_'):
            continue
        name = key[4:]
        if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', name) or name in names:
            raise ValueError(f'{name!r}: variable names must start with a letter, contain letters/digits/underscores and differ from axis names.')
        field = np.asarray(values, dtype=float)
        if field.shape != shape or not np.isfinite(field).all():
            raise ValueError(f'{name}: values must be finite with shape {shape}, got {field.shape}.')
        data[name] = field
    if not data:
        raise ValueError('Supply at least one variable array named var_<name>.')
    return axes, names, data


def custom_identity(path, settings, noise, seed):
    """Content identity for uploads: data, canonical settings, code and dependencies."""
    from epde_bench.identity import _file_hash, _source_hash, _dependency_versions, _digest
    value = {'schema': 1, 'job': {'dataset': 'custom', 'variant': 'custom',
                                 'noise': float(noise), 'seed': int(seed)},
             'config': settings, 'data_sha256': _file_hash(Path(path)),
             'source_sha256': _digest({'bench': _source_hash(),
                                       'custom_run': _file_hash(Path(__file__), source=True)}),
             'dependencies': _dependency_versions()}
    value = json.loads(json.dumps(value, allow_nan=False))
    value['sha256'] = _digest(value)
    return value


def main(argv=None):
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    settings = yaml.safe_load(Path(args.config).read_text(encoding='utf-8')) or {}
    record = {'dataset': 'custom', 'variant': 'custom', 'method': 'epde',
              'noise': float(settings.get('noise', 0.0)), 'seed': int(settings.get('seed', 0)),
              'started': datetime.now(timezone.utc).isoformat(timespec='seconds'),
              'config': settings, 'data_file': str(Path(args.data).resolve())}
    t0 = time.perf_counter()
    try:
        from epde_bench import metrics
        from epde_bench.config import resolve_for_problem
        from epde_bench.runner import discover
        identity = custom_identity(args.data, settings, record['noise'], record['seed'])
        record['identity'] = identity
        record['data_sha256'] = identity['data_sha256']
        record['source'] = {'kind': 'uploaded_npz', 'path': record['data_file'],
                            'sha256': identity['data_sha256']}
        record['environment'] = env.environment_info(PIC_DIR.parent.parent, probe_cuda=False)
        env.seed_everything(record['seed'])
        problem = problem_from_npz(args.data)
        from epde_bench.identity import _file_hash
        if _file_hash(Path(args.data)) != identity['data_sha256']:
            raise ValueError('Uploaded data changed while loading; start a new search.')
        record['problem'] = {'title': problem.title, 'kind': problem.kind, 'shape': list(problem.shape),
                             'variables': problem.variables, 'axis_names': list(problem.axis_names),
                             'truth': None, 'source': problem.source}
        search_cfg = resolve_for_problem({'search': settings.get('search', {})}, problem)
        record['search_config'] = search_cfg
        data = problem.noisy(record['noise'], record['seed'])
        _, texts, objectives, fit_seconds = discover(problem, search_cfg, data)
        record.update(status='ok', fit_seconds=fit_seconds, front=texts, objectives=objectives)
        record['metrics'] = metrics.score_run(texts, objectives, [])
        sel = record['metrics'].get('selected_index')
        record['selected'] = texts[sel] if texts and sel is not None else None
    except Exception as exc:                                    # noqa: BLE001
        record.update(status='error', error=repr(exc), traceback=traceback.format_exc())
    record['total_seconds'] = time.perf_counter() - t0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1, default=str), encoding='utf-8')
    print(record['status'], record.get('selected') or record.get('error'))
    return 0 if record['status'] == 'ok' else 1


if __name__ == '__main__':
    sys.exit(main())
