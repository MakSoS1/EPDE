"""Isolated entry point for PDEComp's original DeepMoD/DISCOVER wrappers."""
import copy
import importlib.util
import json
import sys
import time
from pathlib import Path


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure(config, request):
    method, filename = request['method'], request.get('filename')
    protocol = request.get('protocol', 'native')
    params = copy.deepcopy(getattr(config, method + '_params'))
    if filename is None:
        filename = 'site_data'
        # For new data use the shared polynomial/trigonometric library, with
        # only the target order supplied. No reference coefficients are used.
        kind = request.get('kind', 'ode')
        config.sindy_params[filename] = {
            'targets': request.get('targets', []), 'crop': 2,
            'library': {'type': 'poly_and_fourier', 'derivative_axes': ['t'] if kind.startswith('ode') else ['t', 'x'],
                        'max_deriv_order': [2] if kind.startswith('ode') else [2, 4]},
        }
        params[filename] = ({'direct_optimizer': {'type': 'threshold', 'threshold': 0.1}}
                            if method == 'deepmod' else
                            {'base_config': 'config_pde_Burgers.json', 'function_set': ['add'],
                             'target': (request.get('targets') or [{'name': 'u_t'}])[0]['name']})
    if filename not in params:
        raise ValueError(f'{method} has no PDEComp settings for {filename}')
    settings = {**copy.deepcopy(getattr(config, method.upper() + '_DEFAULTS')), **params[filename]}
    changes = request.get('overrides') or {}
    if protocol == 'native':
        if method == 'deepmod':
            settings = {}
        settings.update(copy.deepcopy(getattr(config, 'NATIVE_' + method.upper() + '_DEFAULTS')))
        settings.update(copy.deepcopy(getattr(config, 'NATIVE_' + method.upper() + '_DATASET_OVERRIDES').get(filename, {})))
        settings.update(changes)
        settings['seed'] = int(request.get('seed', 0))
    elif method == 'deepmod':
        settings['direct_optimizer'] = {**settings.get('direct_optimizer', {}), **changes}
    else:
        settings.update(changes)
        settings['seed'] = int(request.get('seed', 0))
    if method == 'discover' and settings['n_samples'] < settings['batch_size']:
        raise ValueError('DISCOVER n_samples must be at least batch_size')
    if protocol == 'fixed':
        getattr(config, method + '_params')[filename] = settings
    elif method == 'discover' and filename == 'site_data':
        config.discover_params[filename] = params[filename]
    if protocol == 'fixed':
        settings = dict(settings, shared_library=copy.deepcopy(config.sindy_params[filename]),
                        common=copy.deepcopy(config.COMMON_PARAMS))
    else:
        settings = dict(settings, targets=copy.deepcopy(config.sindy_params[filename].get('targets', [{'name': 'u_t', 'axis': 't', 'order': 1}])))
    return filename, settings


def main(request):
    root = Path(request['root'])
    sys.path.insert(0, str(root))
    import data.config as config
    filename, settings = configure(config, request)
    if request.get('describe'):
        Path(request['output']).write_text(json.dumps({'settings': settings}), encoding='utf-8')
        return
    import numpy as np
    method = request['method']
    if method == 'discover':
        load('discover_compat', Path(__file__).with_name('_discover_compat.py')).activate(root)
    else:
        sys.path.insert(0, str(root / 'deepmod' / 'deepymod' / 'src'))
    runner = load('pdecomp_' + method, root / method / 'run.py')
    from utils.randomness import seed_everything
    seed_everything(int(request.get('seed', 0)))
    with np.load(request['input'], allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    data = arrays.get('field')
    if data is None:
        data = [arrays[f'field_{i}'] for i in range(sum(k.startswith('field_') for k in arrays))]
    start = time.perf_counter()
    native_options = {k: v for k, v in settings.items() if k not in ('shared_library', 'common', 'targets')}
    native_options['device'] = 'cpu'
    result = getattr(runner, 'run_' + method)(data, arrays.get('x'), arrays.get('y'), arrays.get('z'),
                                            arrays['t'], filename, protocol=request['protocol'],
                                            native_options=native_options)
    clean = {'targets': list(result['targets']), 'features': [list(f) for f in result['features']],
             'coefficients': [[float(c) for c in np.ravel(cs)] for cs in result['coefficients']]}
    if method == 'discover':
        clean.update({k: result.get(k) for k in ('model', 'discover_reward', 'discover_nmse')})
    Path(request['output']).write_text(json.dumps({'result': clean, 'settings': settings,
                                                  'fit_seconds': time.perf_counter() - start}, default=str),
                                       encoding='utf-8')


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
