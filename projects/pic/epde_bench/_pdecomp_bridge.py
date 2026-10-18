"""Runs PDEComp's PySINDy in an interpreter that sees only PDEComp's packages.

    python _pdecomp_bridge.py '{"root": ..., "filename": ..., "protocol": ..., "seed": ...,
                                "input": "input.npz", "output": "output.json"}'

Started by :func:`epde_bench.pdecomp.run`. The runner is loaded from its file
under a private module name, as PDEComp's own metric script does, so the
installed ``pysindy`` package is not shadowed by PDEComp's folder of that name.
"""

import importlib.util
import json
import sys
import time
from pathlib import Path


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _effective(config, filename, protocol, overrides):
    """PDEComp's settings for one run, with the user's changes applied.

    Fixed protocol: the data set's entry of ``sindy_params`` (optimizer, crop,
    library). Native protocol: PDEComp's native defaults, then its per-data-set
    overrides, then the user's changes, merged exactly as its runner merges them."""
    import copy
    overrides = overrides or {}
    if protocol == 'fixed':
        params = copy.deepcopy(config.sindy_params[filename])
        params['optimizer'] = {**params.get('optimizer', {}), **overrides.get('optimizer', {})}
        if 'crop' in overrides:
            params['crop'] = overrides['crop']
        if overrides.get('library'):
            params['library'] = {**params.get('library', {}), **overrides['library']}
        orders = params.get('library', {}).get('max_deriv_order')
        if orders:
            # The target derivative is taken from the same derivative bundle as the library,
            # so the library must reach the target's order on every axis.
            axes = ['t', 'x', 'y', 'z']
            for target in params.get('targets', []):
                axis = axes.index(target.get('axis', 't')) if target.get('axis', 't') in axes else 0
                if axis < len(orders) and int(target.get('order', 1)) > int(orders[axis]):
                    raise ValueError(f"the highest derivative along {target.get('axis', 't')} ({orders[axis]}) "
                                     f"is below the order of the target {target.get('name')} "
                                     f"({target.get('order', 1)})")
        return params
    dataset = config.NATIVE_PYSINDY_DATASET_OVERRIDES.get(filename, {})
    native = {**config.NATIVE_PYSINDY_DEFAULTS, **copy.deepcopy(dataset)}
    native['optimizer'] = {**config.NATIVE_PYSINDY_DEFAULTS['optimizer'], **dataset.get('optimizer', {}),
                           **overrides.get('optimizer', {})}
    for key in ('polynomial_degree', 'boundary_trim'):
        if key in overrides:
            native[key] = overrides[key]
    if 'differentiation' in overrides:
        native['differentiation'] = {**native.get('differentiation', {}), **overrides['differentiation']}
    return native


def describe(request):
    """The settings a run would use, without running it (for the app's form)."""
    import data.config as config
    filename = request['filename']
    if request.get('params'):
        config.sindy_params[filename] = request['params']
    settings = _effective(config, filename, request['protocol'], request.get('overrides'))
    if request['protocol'] == 'fixed':
        settings = dict(settings, common=dict(config.COMMON_PARAMS))
    Path(request['output']).write_text(json.dumps({'settings': settings}, default=str), encoding='utf-8')


def main(request):
    root = Path(request['root'])
    # Only PDEComp's root: its "data" and "utils" packages must win over anything else.
    sys.path[:] = [str(root)] + [p for p in sys.path if p and Path(p).resolve() != Path(__file__).resolve().parent]
    if request.get('describe'):
        return describe(request)
    import numpy as np
    runner = _load('pdecomp_pysindy_run', root / 'pysindy' / 'run.py')
    metrics = _load('pdecomp_clean_run_metrics', root / 'clean_run_metrics.py')
    from utils.randomness import seed_everything

    with np.load(request['input'], allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    if 'field' in arrays:
        data = arrays['field']
    else:
        data = [arrays[f'field_{i}'] for i in range(sum(k.startswith('field_') for k in arrays))]
    filename = request['filename']
    if request.get('params'):
        # Data outside PDEComp's data sets: only the targets are declared, everything
        # else is PDEComp's native default (no per-data-set overrides exist for it).
        import data.config as config
        config.sindy_params[filename] = request['params']
    import data.config as config
    overrides = request.get('overrides') or {}
    native_options = {'seed': int(request.get('seed', 0)), 'device': 'cpu', 'allow_external_llm': False}
    if request['protocol'] == 'fixed':
        # The runner reads this data set's entry of sindy_params; replace it with the edited one.
        config.sindy_params[filename] = effective = _effective(config, filename, 'fixed', overrides)
    else:
        native = effective = _effective(config, filename, 'native', overrides)
        native_options.update({k: native[k] for k in ('optimizer', 'polynomial_degree', 'boundary_trim',
                                                      'differentiation') if k in native})
    seed_everything(int(request.get('seed', 0)))
    t0 = time.perf_counter()
    result = runner.run_sindy(data, arrays.get('x'), arrays.get('y'), arrays.get('z'), arrays['t'],
                              filename, protocol=request['protocol'], native_options=native_options)
    fit_seconds = time.perf_counter() - t0

    # PDEComp's own structural metrics: Hamming distance and the sum of relative
    # coefficient errors over the true terms, per target and for the system.
    per_target = []
    for i, target in enumerate(result['targets']):
        fitted = metrics.coefficient_by_feature(result, i)
        match = metrics.best_true_coefficient_match(filename, target, fitted)
        per_target.append({'target': target,
                           'hd': None if match is None else int(match['hamming']),
                           're': None if match is None else float(match['relative_error_sum'])})
    known = [row for row in per_target if row['hd'] is not None]
    summary = {'per_target': per_target,
               'hd': sum(r['hd'] for r in known) if known else None,
               're': sum(r['re'] for r in known) if known else None}
    clean = {'targets': list(result['targets']),
             'features': [list(map(str, f)) for f in result['features']],
             'coefficients': [[float(c) for c in np.ravel(cs)] for cs in result['coefficients']],
             'protocol': result.get('protocol', request['protocol'])}
    Path(request['output']).write_text(json.dumps({'result': clean, 'metrics': summary,
                                                   'fit_seconds': fit_seconds,
                                                   'settings': effective}, default=str),
                                       encoding='utf-8')


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
