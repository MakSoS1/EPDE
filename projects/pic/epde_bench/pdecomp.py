"""PySINDy exactly as the PDEComp benchmark runs it, on the data of this project.

PDEComp (github.com/emil817/PDEComp) compares equation-discovery frameworks
under two protocols: ``fixed`` (shared derivatives and candidate library, a
tuned optimizer per data set) and ``native`` (PySINDy's own differentiation and
libraries). This module runs PDEComp's own PySINDy code, pinned to one commit,
on the arrays a search here receives, and converts its result into the EPDE text
form used by the metrics and the app.

The PDEComp source is not copied into this repository. The few files the
runner imports are downloaded once from the pinned commit, checked against
their SHA-256 and cached; ``PDECOMP_DIR`` may point to a local checkout of the
same commit instead. The runner executes in a separate interpreter because
PDEComp's top-level packages (``data``, ``utils``) would shadow names here.

On PDEComp's own data sets (:data:`PDECOMP_FILES`, the same files, windows and
grids as PDEComp) both protocols use PDEComp's settings for that data set. Any
other record, uploaded file or generated ODE runs under the ``native`` protocol
with PDEComp's default settings; like PDEComp's own data sets it is given only
the target derivative (taken from the left-hand side of the known law, else
first order in time). The native protocol covers ODEs, systems of ODEs and
scalar PDEs in one space dimension.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np

from .paths import PIC_DIR

PDECOMP_REPO = 'emil817/PDEComp'
PDECOMP_REV = '4abfe4e9eef5263c35ae5672c6ebfb65aabf8365'
PROTOCOLS = ('fixed', 'native')

#: Files the PySINDy runner and the metric functions import, with SHA-256 at the pinned commit.
SOURCE_FILES = {
    'pysindy/run.py': '7f1b6c8aacfe05c667a3abb20ec27d5b9b40e1d0f27db0a04bc0b2bca9c6e7be',
    'utils/__init__.py': '890d322f67389fea07c3c29f13e228473df3cca49f7d7eba6f2667e94691a458',
    'utils/sindy_library.py': '2ded5ce3e92ea95ad4e049b2651f30ab6b2cc48f8fe2504add9144aa08899e55',
    'utils/derivatives.py': '5e6520b458498ff9491b5a009074f1c52ec851b171887f424740315c285115b2',
    'utils/protocols.py': '720d8e80e5fbf4a933bb452c8ee78bed4e6905cbdc191bcc964320cfe6ec7e4e',
    'utils/dataloader.py': '22967acc05f1409cf6456a951055299f5c5b59fb2436ed6df849d70a5dd4ecee',
    'utils/randomness.py': '8e318030ba10e647e95afaf65ea229aa0c1b1fdaae234c7fd3e173fae10430a9',
    'data/config.py': '6e15b84b2d29297a74b91d14ce1367c5083a718d4f58f377881776b7742afe3e',
    'clean_run_metrics.py': '4230c4a034e0f148bc7f92fe341d46e7b2f6dfe8b1bb728c84d742f15335ad40',
}

#: Records of this project whose arrays are PDEComp's data sets, and their PDEComp names.
PDECOMP_FILES = {
    'ode': 'ode_data.npy',
    'vdp': 'vdp_data.npy',
    'pdecomp_lv': 'lotka_data.npy',
    'pdecomp_lorenz': 'lorenz_data.npy',
    'burgers': 'burgers_data.mat',
    'burgers_inviscid': 'burgers_sln_100_data.csv',
    'ac': 'ac_data.npy',
    'kdv': 'kdv_data.mat',
    'kdv_cossin': 'kdv_periodic_data.npy',
    'wave': 'wave_data.csv',
    'pdecomp_pde_divide': 'pde_divide_data.npy',
    'pde_compound': 'pde_compound_data.npy',
    'ks': 'ks_data.mat',
    'pdecomp_ns': 'ns_data.mat',
    'pdecomp_ode_simple': 'ODE_simple_discovery',
}

CACHE_DIR = PIC_DIR / '.cache' / 'pdecomp'
BRIDGE = Path(__file__).with_name('_pdecomp_bridge.py')


class PdecompUnavailable(RuntimeError):
    """The pinned PDEComp source could not be obtained."""


def supported(dataset: str) -> bool:
    """A PDEComp data set, with PDEComp's own per-data-set settings."""
    return dataset in PDECOMP_FILES


def unsupported_reason(kind: str, variables: int):
    """Why the native protocol cannot represent data of this shape, or None."""
    if kind in ('ode', 'ode_system'):
        return None
    if kind == 'pde_1d' and variables == 1:
        return None
    if kind == 'pde_1d':
        return 'PDEComp\'s PySINDy handles one variable in one space dimension'
    return 'PDEComp\'s PySINDy handles PDEs in one space dimension (Navier-Stokes only as its own data set)'


def protocols_for(dataset):
    """Protocols available for a record: both on PDEComp's data sets, native elsewhere."""
    return PROTOCOLS if supported(dataset) else ('native',)


_TARGET_RE = re.compile(r'^d(?:\^(\d+))?([A-Za-z_][A-Za-z0-9_]*)/dx0(?:\^\d+)?\{power: 1(?:\.0)?\}$')


def target_orders(problem, default: int = 1):
    """Order of the time derivative each variable's law is solved for.

    Read from the left-hand side of the known law when it is a pure time
    derivative; otherwise (or with no known law) ``default``."""
    orders = {v: default for v in problem.variables}
    for eq in problem.truth or []:
        target = eq.split('=', 1)[1].strip()
        m = _TARGET_RE.match(target)
        if m and m.group(2) in orders:
            orders[m.group(2)] = int(m.group(1) or 1)
    return orders


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _complete(root: Path) -> bool:
    return all((root / name).is_file() and _sha256(root / name) == digest
               for name, digest in SOURCE_FILES.items())


def source_dir(download: bool = True) -> Path:
    """A directory with the pinned PDEComp files, verified by checksum."""
    override = os.environ.get('PDECOMP_DIR')
    if override:
        root = Path(override)
        if not _complete(root):
            raise PdecompUnavailable(f'PDECOMP_DIR={root} is not PDEComp at commit {PDECOMP_REV[:7]}')
        return root
    root = CACHE_DIR / PDECOMP_REV
    if _complete(root):
        return root
    if not download:
        raise PdecompUnavailable('PDEComp source is not cached yet')
    for name, digest in SOURCE_FILES.items():
        target = root / name
        if target.is_file() and _sha256(target) == digest:
            continue
        url = f'https://raw.githubusercontent.com/{PDECOMP_REPO}/{PDECOMP_REV}/{name}'
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                payload = response.read(2_000_000)
        except OSError as exc:
            raise PdecompUnavailable(f'cannot download {name} of PDEComp: {exc}') from exc
        if hashlib.sha256(payload).hexdigest() != digest:
            raise PdecompUnavailable(f'{name} of PDEComp does not match the pinned checksum')
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + '.part')
        tmp.write_bytes(payload)
        tmp.replace(target)
    return root


def _pdecomp_axes(ndim):
    """PDEComp's axis names by position: (t,), (t, x) or (t, y, x)."""
    return {1: ['t'], 2: ['t', 'x'], 3: ['t', 'y', 'x']}[ndim]


def _layout(problem, data):
    """Arrays in the order PDEComp's runner expects: ``data, x, y, z, t``."""
    axes = _pdecomp_axes(len(problem.axis_names))

    def axis(name):
        if name not in axes:
            return None
        i = axes.index(name)
        grid = np.asarray(problem.grids[i])
        if grid.ndim == 1:
            return grid
        index = [0] * grid.ndim
        index[i] = slice(None)
        return grid[tuple(index)]

    fields = [np.asarray(data[v], dtype=float) for v in problem.variables]
    scalar = len(fields) == 1 and PDECOMP_FILES.get(problem.name) != 'ODE_simple_discovery'
    return (fields[0] if scalar else fields), axis('x'), axis('y'), axis('z'), axis('t')


#: Settings of PDEComp's PySINDy that the app lets the user change, per protocol.
OPTIMIZER_KEYS = {'type', 'threshold', 'alpha', 'max_iter', 'normalize_columns',
                  'coefficient_tol', 'coefficient_relative_tol'}
OPTIMIZERS = ('STLSQ', 'FROLS')     # PDEComp's SR3 branch does not run with PySINDy 2.1
#: Extra library terms PDEComp's fixed protocol can build, by the kind of data.
TIME_TOKENS = ('t', 't^2', 'sin(t)', 'cos(t)', 'sin(2 t)', 'cos(2 t)')
CUSTOM_TOKENS = {
    'ode': TIME_TOKENS + ('u_t sin(2 t)',),
    'ode_system': TIME_TOKENS,
    'pde_1d': TIME_TOKENS + ('x', 'u_t sin(2 t)', 'sin(x)', 'sin(x) cos(t)', 'cos(x) sin(t)',
                             '(1/x) u', '(1/x) u_x', 'd_x(u u_x)'),
}


def validate_overrides(overrides, protocol, kind=None):
    """Check user changes to PDEComp's settings; returns a clean copy or raises."""
    if overrides in (None, {}):
        return {}
    if not isinstance(overrides, dict):
        raise ValueError('PySINDy settings must be a mapping')
    allowed = {'optimizer'} | ({'crop', 'library'} if protocol == 'fixed'
                               else {'polynomial_degree', 'boundary_trim', 'differentiation'})
    if set(overrides) - allowed:
        raise ValueError('Unknown PySINDy setting: ' + ', '.join(sorted(set(overrides) - allowed)))
    out = {}

    def number(value, low, high, name, integer=False):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
            raise ValueError(f'{name} must be between {low:g} and {high:g}')
        if integer and int(value) != value:
            raise ValueError(f'{name} must be an integer')
        return int(value) if integer else float(value)

    opt = overrides.get('optimizer') or {}
    if not isinstance(opt, dict) or set(opt) - OPTIMIZER_KEYS:
        raise ValueError('Unknown optimizer setting')
    clean = {}
    if 'type' in opt:
        if opt['type'] not in OPTIMIZERS:
            raise ValueError('The optimizer must be STLSQ or FROLS')
        clean['type'] = opt['type']
    for key, low, high, integer in (('threshold', 0.0, 1e4, False), ('alpha', 0.0, 1e4, False),
                                    ('max_iter', 1, 200, True), ('coefficient_tol', 0.0, 1e4, False),
                                    ('coefficient_relative_tol', 0.0, 1.0, False)):
        if key in opt:
            clean[key] = number(opt[key], low, high, key, integer)
    if 'normalize_columns' in opt:
        if not isinstance(opt['normalize_columns'], bool):
            raise ValueError('normalize_columns must be true or false')
        clean['normalize_columns'] = opt['normalize_columns']
    if clean:
        out['optimizer'] = clean
    if 'crop' in overrides:
        out['crop'] = number(overrides['crop'], 0, 50, 'crop', True)
    if 'library' in overrides:
        library = overrides['library'] or {}
        keys = {'data_fun_pow', 'equation_factors_max_number', 'include_bias', 'max_deriv_order', 'custom_tokens'}
        if not isinstance(library, dict) or set(library) - keys:
            raise ValueError('Unknown library setting: ' + ', '.join(sorted(set(library) - keys)))
        clean = {}
        if 'data_fun_pow' in library:
            clean['data_fun_pow'] = number(library['data_fun_pow'], 1, 4, 'polynomial degree', True)
        if 'equation_factors_max_number' in library:
            clean['equation_factors_max_number'] = number(library['equation_factors_max_number'], 1, 2,
                                                          'factors per term', True)
        if 'include_bias' in library:
            if not isinstance(library['include_bias'], bool):
                raise ValueError('include_bias must be true or false')
            clean['include_bias'] = library['include_bias']
        if 'max_deriv_order' in library:
            orders = library['max_deriv_order']
            if not isinstance(orders, (list, tuple)) or not 1 <= len(orders) <= 2:
                raise ValueError('max_deriv_order is (time, space)')
            clean['max_deriv_order'] = [number(orders[0], 1, 2, 'time derivative order', True)] + \
                [number(o, 1, 4, 'space derivative order', True) for o in orders[1:]]
        if 'custom_tokens' in library:
            tokens = library['custom_tokens']
            vocabulary = set().union(*CUSTOM_TOKENS.values()) if kind is None else set(CUSTOM_TOKENS.get(kind, ()))
            if not isinstance(tokens, (list, tuple)) or any(t not in vocabulary for t in tokens):
                raise ValueError('Unknown extra library term for these data')
            clean['custom_tokens'] = list(dict.fromkeys(tokens))
        if clean:
            out['library'] = clean
    if 'polynomial_degree' in overrides:
        out['polynomial_degree'] = number(overrides['polynomial_degree'], 1, 4, 'polynomial degree', True)
    if 'boundary_trim' in overrides:
        out['boundary_trim'] = number(overrides['boundary_trim'], 0, 50, 'boundary trim', True)
    if 'differentiation' in overrides:
        diff = overrides['differentiation'] or {}
        if not isinstance(diff, dict) or set(diff) - {'order', 'periodic'}:
            raise ValueError('Unknown differentiation setting')
        clean = {}
        if 'order' in diff:
            if diff['order'] not in (2, 4):
                raise ValueError('The finite-difference order must be 2 or 4')
            clean['order'] = int(diff['order'])
        if 'periodic' in diff:
            if not isinstance(diff['periodic'], bool):
                raise ValueError('periodic must be true or false')
            clean['periodic'] = diff['periodic']
        if clean:
            out['differentiation'] = clean
    return out


def _bridge(request, timeout):
    root = source_dir()
    env = dict(os.environ, PYTHONIOENCODING='utf-8', MPLBACKEND='Agg')
    env.pop('PYTHONPATH', None)
    request = dict(request, root=str(root))
    proc = subprocess.run([sys.executable, str(BRIDGE), json.dumps(request)],
                          capture_output=True, text=True, timeout=timeout, env=env, cwd=str(root))
    if proc.returncode != 0:
        lines = (proc.stderr or proc.stdout or '').strip().splitlines()
        if lines and lines[-1].startswith('ValueError: '):
            raise ValueError('PySINDy settings: ' + lines[-1][len('ValueError: '):])
        raise RuntimeError('PDEComp PySINDy failed: ' + ' | '.join(lines[-5:]))
    return json.loads(Path(request['output']).read_text(encoding='utf-8'))


def describe(dataset, protocol, overrides=None):
    """The settings PDEComp's PySINDy uses for a record (``None`` = data outside
    PDEComp's data sets), with ``overrides`` applied; nothing is run."""
    filename = PDECOMP_FILES.get(dataset)
    if filename is None and protocol != 'native':
        raise ValueError('the fixed protocol has settings only for PDEComp\'s data sets')
    with tempfile.TemporaryDirectory() as tmp:
        reply = _bridge({'describe': True, 'filename': filename or 'site:data', 'protocol': protocol,
                         'params': None if filename else {'targets': []},
                         'overrides': validate_overrides(overrides, protocol),
                         'output': str(Path(tmp) / 'settings.json')}, timeout=120)
    return reply['settings']


def run(problem, data, protocol: str = 'fixed', seed: int = 0, timeout: float = 600.0,
        orders=None, overrides=None):
    """Run PDEComp's PySINDy once.

    Returns ``(raw_result, pdecomp_metrics, seconds, effective_settings)``.
    ``orders`` ({variable: 1 or 2}) sets the target time derivatives of data
    outside PDEComp's data sets; by default they come from :func:`target_orders`.
    ``overrides`` are the user's changes to PDEComp's settings (see
    :func:`validate_overrides`)."""
    if protocol not in PROTOCOLS:
        raise ValueError(f'unknown PDEComp protocol {protocol!r}')
    overrides = validate_overrides(overrides, protocol, problem.kind)
    filename = PDECOMP_FILES.get(problem.name)
    params = None
    if filename is None:
        if protocol != 'native':
            raise ValueError('the fixed protocol has settings only for PDEComp\'s data sets')
        reason = unsupported_reason(problem.kind, len(problem.variables))
        if reason:
            raise ValueError(reason)
        orders = dict(target_orders(problem), **(orders or {}))
        if len(problem.variables) > 1 and any(k != 1 for k in orders.values()):
            raise ValueError('PDEComp\'s PySINDy solves systems for first time derivatives only')
        if any(k not in (1, 2) for k in orders.values()):
            raise ValueError('the target time derivative must be of order 1 or 2')
        names = ['u'] if len(problem.variables) == 1 else [f'x{i}' for i in range(len(problem.variables))]
        targets = [{'name': f"{n}_{'t' * orders[v]}", 'variable': n, 'axis': 't', 'order': orders[v]}
                   for n, v in zip(names, problem.variables)]
        filename = f'site:{problem.name}'
        params = {'targets': targets}
    fields, x, y, z, t = _layout(problem, data)
    with tempfile.TemporaryDirectory() as tmp:
        arrays = {'t': t}
        for name, value in (('x', x), ('y', y), ('z', z)):
            if value is not None:
                arrays[name] = value
        if isinstance(fields, list):
            arrays.update({f'field_{i}': f for i, f in enumerate(fields)})
        else:
            arrays['field'] = fields
        np.savez(Path(tmp) / 'input.npz', **arrays)
        request = {'filename': filename, 'protocol': protocol, 'params': params,
                   'overrides': overrides, 'seed': int(seed),
                   'input': str(Path(tmp) / 'input.npz'), 'output': str(Path(tmp) / 'output.json')}
        t0 = time.perf_counter()
        reply = _bridge(request, timeout)
        seconds = time.perf_counter() - t0
    return reply['result'], reply['metrics'], reply.get('fit_seconds', seconds), reply.get('settings')


# ---------------------------------------------------------------------------
# PDEComp feature names -> EPDE text
# ---------------------------------------------------------------------------

_TRIG_RE = re.compile(r'^(sin|cos)\((?:([0-9.]+)\s*)?([txyz])\)$')
_NAME_RE = re.compile(r'^([A-Za-z][A-Za-z0-9]*)(?:_([txyz]+))?(?:\^(\d+))?$')


def _split_factors(text):
    """Split on spaces outside parentheses: ``'u_t sin(2 t)'`` -> ``['u_t', 'sin(2 t)']``."""
    out, depth, current = [], 0, ''
    for ch in text.strip():
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        if ch == ' ' and depth == 0:
            if current:
                out.append(current)
            current = ''
        else:
            current += ch
    if current:
        out.append(current)
    return out


def _number(value):
    return f'{float(value):.1f}' if float(value).is_integer() else repr(float(value))


class _Translator:
    """Maps PDEComp's names to EPDE tokens for one problem."""

    #: PDEComp products that a record of this project declares as one token.
    PRODUCT_TOKENS = {'sin(x) cos(t)': 'cos(t)sin(x)', 'cos(t) sin(x)': 'cos(t)sin(x)',
                      'cos(x) sin(t)': 'sin(t)cos(x)', 'sin(t) cos(x)': 'sin(t)cos(x)'}

    def __init__(self, problem):
        self.axes = _pdecomp_axes(len(problem.axis_names))
        self.variables = list(problem.variables)
        self.extra = set(problem.extra_arrays or {})

    def variable(self, name):
        if name in self.variables:
            return name
        m = re.fullmatch(r'x(\d+)', name)
        if m and int(m.group(1)) < len(self.variables):
            return self.variables[int(m.group(1))]
        if name == 'u' and len(self.variables) == 1:
            return self.variables[0]
        raise KeyError(name)

    def factor(self, text):
        """``{token_text_without_power: power}`` for one PDEComp factor, or a list of
        such dicts when the factor is a sum (``d_x(u u_x)``, ``(u_xx + u_yy)``)."""
        if text == '1':
            return {}
        if text == '(1/x)':
            return {'1/x': 1}
        if text.startswith('d_x(') and text.endswith(')'):
            inner = _split_factors(text[4:-1])
            if inner == ['u', 'u_x']:
                v = self.variable('u')
                d1, d2 = self.derivative(v, 'x'), self.derivative(v, 'xx')
                return [{d1: 2}, {v: 1, d2: 1}]
            raise KeyError(text)
        if text.startswith('(') and text.endswith(')') and ' + ' in text:
            return [self.factor(part.strip()) for part in text[1:-1].split(' + ')]
        m = _TRIG_RE.match(text)
        if m:
            fn, freq, axis = m.group(1), float(m.group(2) or 1.0), m.group(3)
            return {f'{fn}{{power: 1.0, freq: {_number(freq)}, dim: {self.axes.index(axis)}.0}}': 1}
        m = _NAME_RE.match(text)
        if not m:
            raise KeyError(text)
        name, suffix, power = m.group(1), m.group(2), int(m.group(3) or 1)
        if suffix is None and name in self.axes:
            return {f'x{{power: {{p}}, dim: {self.axes.index(name)}.0}}': power}
        variable = self.variable(name)
        token = variable if suffix is None else self.derivative(variable, suffix)
        return {token: power}

    def derivative(self, variable, suffix):
        if len(set(suffix)) != 1:
            raise KeyError(f'mixed derivative {variable}_{suffix}')
        axis, order = self.axes.index(suffix[0]), len(suffix)
        return f'd{variable}/dx{axis}' if order == 1 else f'd^{order}{variable}/dx{axis}^{order}'

    def term(self, feature):
        """A PDEComp feature as a list of products ``{token: power}`` (sum of products)."""
        products = [{}]
        factors = _split_factors(feature)
        for i in range(len(factors) - 1):
            label = self.PRODUCT_TOKENS.get(f'{factors[i]} {factors[i + 1]}')
            if label in self.extra:
                factors[i:i + 2] = [f'<{label}>']
                break
        for text in factors:
            if text.startswith('<') and text.endswith('>'):
                products = [dict(p, **{text[1:-1]: p.get(text[1:-1], 0) + 1}) for p in products]
                continue
            part = self.factor(text)
            options = part if isinstance(part, list) else [part]
            expanded = []
            for product in products:
                for option in options:
                    merged = dict(product)
                    for token, power in option.items():
                        merged[token] = merged.get(token, 0) + power
                    expanded.append(merged)
            products = expanded
        return products

    @staticmethod
    def text(product):
        parts = []
        for token, power in sorted(product.items()):
            if '{p}' in token:
                parts.append(token.replace('{p}', _number(power)))
            elif token.startswith(('sin{', 'cos{')):
                parts.append(token.replace('power: 1.0', f'power: {_number(power)}', 1))
            else:
                parts.append(f'{token}{{power: {_number(power)}}}')
        return ' * '.join(parts)


def to_epde_texts(problem, result, tolerance: float = 1e-12):
    """One EPDE equation string per PDEComp target, in the target order."""
    tr = _Translator(problem)
    equations = []
    for target, coefficients, features in zip(result['targets'], result['coefficients'],
                                              result['features']):
        sums = {}
        constant = 0.0
        for feature, coef in zip(features, coefficients):
            coef = float(coef)
            if abs(coef) <= tolerance:
                continue
            for product in tr.term(feature):
                if not product:
                    constant += coef
                    continue
                key = tr.text(product)
                sums[key] = sums.get(key, 0.0) + coef
        terms = [f'{coef!r} * {key}' for key, coef in sums.items() if abs(coef) > tolerance]
        if constant:
            terms.append(repr(constant))
        lhs_product = tr.term(target)
        if len(lhs_product) != 1:
            raise KeyError(target)
        equations.append(' + '.join(terms or ['0.0']) + ' = ' + tr.text(lhs_product[0]))
    return equations


def library_size(result):
    return [len(features) for features in result['features']]
