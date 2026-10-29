"""Interpret saved runs without fitting models or loading numerical datasets.

Leaders describe selected configurations on one verified input. Internal
optimizer scores and oracle best-front candidates are deliberately not ranked.
"""
from __future__ import annotations

import hashlib
import json
import math
import re

from .portable import input_key


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _number(value, *, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or (nonnegative and value < 0):
        return None
    return float(value)


def _signature(value):
    try:
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                         allow_nan=False).encode()).hexdigest()
    except (TypeError, ValueError):
        return None


def _input(record):
    try:
        return input_key(record)
    except (TypeError, ValueError, AttributeError, KeyError):
        return None


def _variables(record):
    variables = (_mapping(record.get('problem')).get('variables')
                 or _mapping(record.get('evaluation')).get('variables') or [])
    return list(variables) if isinstance(variables, (list, tuple, dict)) else []


def _worst_r2(scores, variables):
    scores = _mapping(scores)
    values = [_number(scores.get(var)) for var in (variables or list(scores))]
    if not values or any(value is None or value > 1 for value in values):
        return None
    return min(values)


def _solution(record):
    evaluation = _mapping(record.get('evaluation'))
    solution = _mapping(evaluation.get('solution'))
    if solution.get('success') is not True or _number(solution.get('coverage')) != 1:
        return None, None
    variables = _variables(record)
    value = _worst_r2(solution.get('r2'), variables)
    observed = _mapping(solution.get('observed'))
    # Same input does not imply the same boundary cuts or evaluation interval.
    if not solution.get('t') or not observed or any(v not in observed for v in variables):
        return value, None
    basis = _signature({'kind': evaluation.get('kind'), 'variables': variables,
                        't': solution['t'], 'x': solution.get('x'), 'observed': observed})
    return value, basis


def _heldout(record):
    # Existing epde_bench.heldout schema; evaluation.solution is in-sample.
    protocol = record.get('protocol')
    train, test = _mapping(record.get('train')), _mapping(record.get('test'))
    if (not isinstance(protocol, str) or 'forecast' not in protocol.lower()
            or not train.get('data_sha256') or not test.get('time_range')
            or test.get('integration_success') is not True
            or _number(test.get('coverage')) != 1
            or not isinstance(record.get('split_index'), int)
            or not isinstance(record.get('total_points'), int)
            or not isinstance(test.get('n_points'), int)
            or test['n_points'] != record['total_points'] - record['split_index']):
        return None, None
    scores = {var: _mapping(metrics).get('r2')
              for var, metrics in _mapping(test.get('metrics')).items()}
    basis = _signature({'protocol': protocol, 'train': train['data_sha256'],
                        'split': record['split_index'], 'total': record['total_points'],
                        'n': test['n_points'], 'range': test['time_range'],
                        'variables': _variables(record)})
    return _worst_r2(scores, _variables(record)), basis


def _environment(record):
    env = _mapping(record.get('environment'))
    required = ('platform', 'machine', 'python', 'cpu_count', 'blas_threads')
    if any(env.get(key) in (None, '', 'unknown') for key in required):
        return None
    if type(env.get('cuda')) is not bool:
        return None
    fields = {key: env.get(key) for key in (*required, 'cuda', 'numpy', 'scipy', 'torch', 'pysindy')}
    fields['location'] = record.get('execution_location')
    fields['dependencies'] = _mapping(record.get('identity')).get('dependencies')
    device = _mapping(_mapping(record.get('search_config')).get('solver')).get('device')
    fields['device'] = device if device not in (None, 'auto') else ('cuda available' if env['cuda'] else 'cpu')
    return _signature(fields)


def _protocol(record):
    method = record.get('method')
    return (_mapping(record.get('baseline')).get('protocol')
            or _mapping(_mapping(record.get('config')).get(method)).get('protocol')
            or (record.get('variant') if method == 'epde' else None) or 'not recorded')


def _reference(record):
    problem, config = _mapping(record.get('problem')), _mapping(record.get('config'))
    truth = problem.get('truth')
    if not truth:
        return None
    method_config = _mapping(config.get(record.get('method')))
    # Native adapters infer omitted target orders from the declared reference.
    # Treat an explicit equivalent order as the same evaluation definition.
    targets = {var: 1 for var in _variables(record)}
    for equation in truth:
        if isinstance(equation, str) and '=' in equation:
            match = re.fullmatch(r'd(?:\^(\d+))?([A-Za-z_][A-Za-z0-9_]*)/dx0(?:\^\d+)?'
                                 r'\{power: 1(?:\.0)?\}', equation.split('=', 1)[1].strip())
            if match and match[2] in targets:
                targets[match[2]] = int(match[1] or 1)
    targets.update(_mapping(method_config.get('target_orders')))
    return _signature({'truth': truth, 'alternatives': problem.get('truth_alternatives'),
                        'alternative_count': problem.get('n_truth_alternatives'),
                        'variables': _variables(record),
                        'targets': targets})


def _run(label, record):
    metrics, baseline = _mapping(record.get('metrics')), _mapping(record.get('baseline'))
    truth = _mapping(record.get('problem')).get('truth')
    status = record.get('status', 'unknown')
    scorable = (status == 'ok' and bool(record.get('selected')) and bool(truth)
                and not (metrics.get('scoring_note') or baseline.get('scoring_note')))
    hd = _number(metrics.get('hamming_selected'), nonnegative=True) if scorable else None
    if hd is not None and not hd.is_integer():
        hd = None
    selected = metrics.get('success_selected') if scorable else None
    if hd is not None and type(selected) is bool and (hd == 0) != selected:
        hd, scorable, selected = None, False, None
    re = _number(metrics.get('re_sum'), nonnegative=True) if scorable else None
    # coef_error and best_index concern an oracle candidate, not the selected law.
    law = ('Law unknown' if not truth else 'Not scored' if not scorable else
           'Exact structure' if hd == 0 or selected is True else
           'Different structure' if hd is not None or selected is False else 'Not scored')
    solution, solution_basis = _solution(record) if status == 'ok' else (None, None)
    heldout, heldout_basis = _heldout(record) if status == 'ok' else (None, None)
    seconds = _number(record.get('fit_seconds'), nonnegative=True) if status == 'ok' else None
    advice = []
    if status == 'timeout':
        budget = ('training iterations or sample budget' if record.get('method') == 'deepmod' else
                  'expression sample budget' if record.get('method') == 'discover' else 'search budget')
        advice.append(f'Reduce the {budget}, or run the same experiment locally with more time. '
                      'A timeout does not establish equation quality.')
    elif status == 'unsupported':
        advice.append('Choose a compatible data shape or protocol in Settings. A larger budget '
                      'does not resolve an unsupported combination.')
    elif status != 'ok':
        advice.append('Open this run in Results and resolve the execution error before changing '
                      'scientific settings or comparing its quality.')
    elif metrics.get('scoring_note') or baseline.get('scoring_note'):
        advice.append('Inspect the symbolic equation in Results: shared structural scoring is '
                      'unavailable for this expression. Missing scores do not mean zero error.')
    else:
        if metrics.get('success_front') is True and law == 'Different structure':
            advice.append('The correct structure occurred among candidates but was not selected. '
                          'Inspect candidate selection using independent validation; increasing '
                          'the search budget alone may not resolve this difference.')
        elif law == 'Different structure':
            advice.append('Repeat several seeds with the same settings. If the structural mismatch '
                          'persists, review derivative settings, noise and the allowed feature library.')
        elif law == 'Exact structure' and re is not None and re > 0:
            advice.append('The selected structure matches. To improve coefficients, check data '
                          'scaling, derivative accuracy and noise sensitivity across repeated runs.')
        if solution is not None and solution >= 0.95 and law == 'Different structure':
            advice.append('A close in-sample trajectory can still come from a different equation. '
                          'Check a held-out trajectory or different initial conditions.')
    saved_solution = _mapping(_mapping(record.get('evaluation')).get('solution'))
    if status == 'ok' and saved_solution and (saved_solution.get('success') is not True
                                             or _number(saved_solution.get('coverage')) != 1):
        advice.append('Solution reconstruction did not cover the full interval. Check integration '
                      'stability and initial/boundary conditions before interpreting its fit.')
    elif status == 'ok' and solution is None:
        advice.append('Solution fit is unavailable. Check the reconstruction note in Results; '
                      'this may reflect an unsupported equation form rather than poor discovery.')
    if status == 'ok' and heldout is None:
        advice.append('Validate on held-out observations or a new trajectory before making '
                      'a prediction claim; the displayed solution fit uses the search data.')
    return {'label': str(label), 'method': record.get('method', 'unknown'),
            'protocol': _protocol(record), 'status': status, 'selected_law': law,
            'hd': int(hd) if hd is not None else None, 're': re,
            'solution_r2': solution, 'heldout_r2': heldout, 'seconds': seconds,
            'advice': advice, '_truth_basis': _reference(record),
            '_solution_basis': solution_basis, '_heldout_basis': heldout_basis,
            '_time_basis': _environment(record)}


def _metric(rows, verified, key, title, direction, basis_key, explanation, incompatibility):
    available = [row for row in rows if row[key] is not None]
    bases = {row[basis_key] for row in available}
    leaders, value = [], None
    if not verified:
        reason = 'Input or scientific-source fingerprints differ or are missing; no leader is established.'
    elif len(available) < 2:
        reason = f'Available for {len(available)} of {len(rows)} selected runs; at least two comparable scores are needed.'
    elif None in bases or len(bases) != 1:
        reason = incompatibility
    else:
        value = (min if direction == 'min' else max)(row[key] for row in available)
        leaders = [row['label'] for row in available
                   if math.isclose(row[key], value, rel_tol=1e-9, abs_tol=1e-12)]
        reason = f'Compared {len(available)} of {len(rows)} selected runs. '
        if len(available) < len(rows):
            reason += 'Unavailable scores are excluded, never replaced by zero. '
        reason += explanation
    return {'key': key, 'title': title, 'direction': direction, 'leaders': leaders,
            'value': value, 'eligible': len(available), 'total': len(rows), 'explanation': reason}


def analyze_comparison(runs):
    """Return per-metric leaders, coverage notes and next steps for saved runs.

    Input is ``[(display_label, record), ...]``. Grouping is verified again here;
    stored results are interpreted rather than independently recomputed.
    """
    runs = [(label, _mapping(record)) for label, record in runs]
    keys = [_input(record) for _, record in runs]
    verified = bool(keys) and None not in keys and len(set(keys)) == 1
    rows = [_run(label, record) for label, record in runs]
    notes = ['These highlights describe the selected settings on this input. Internal optimizer '
             'objectives are not a shared quality scale, so there is no combined algorithm score.']
    if not verified:
        notes.append('Matching data, source, loader, noise and seed fingerprints are required '
                     'before highlighting a leader.')
    if len({(row['method'], row['protocol']) for row in rows}) > 1:
        notes.append('Methods and protocols use different libraries and derivative estimators. '
                     'Common output metrics compare complete configured pipelines, not an isolated optimizer.')
    configurations = {_signature({'search': record.get('search_config'),
                                  'method': _mapping(record.get('config')).get(record.get('method')),
                                  'settings': _mapping(record.get('baseline')).get('settings')})
                      for _, record in runs}
    if len(configurations) > 1:
        notes.append('Recorded search configurations differ. Time highlights describe those '
                     'particular budgets and settings; they do not measure equal-effort algorithm speed.')
    if not any(row['heldout_r2'] is not None for row in rows):
        notes.append('No independent held-out forecast is recorded for these runs. Solution R² '
                     'is an in-sample reconstruction, not forecast accuracy.')
    if any(record.get('execution_location') == 'user_computer' for _, record in runs):
        notes.append('Imported results are checked for internal consistency; the website has '
                     'not independently repeated those experiments.')
    notes.append('Repeat seeds and noise levels before generalizing from individual runs.')
    definitions = [
        ('hd', 'Selected structure · HD', 'min', '_truth_basis',
         'Lower is better; zero means the selected equation has the reference structure.',
         'Reference laws or target definitions differ or are missing; structural scores are not ranked.'),
        ('re', 'Selected coefficients · RE', 'min', '_truth_basis',
         'Lower is better on the reference terms: a missing true term adds 1; extra terms are counted by HD. This scores the selected equation, not an oracle best-front candidate.',
         'Reference laws or target definitions differ or are missing; coefficient scores are not ranked.'),
        ('solution_r2', 'In-sample solution R²', 'max', '_solution_basis',
         'Higher is better on the same recorded reconstruction grid; the worst variable is shown. This is not a forecast.',
         'Reconstruction grids or observations differ or are not recorded; solution-fit leaders are unavailable.'),
        ('seconds', 'Recorded search time', 'min', '_time_basis',
         'Same reported environment. Budgets and host load may differ; this is observed time for these settings, not an algorithm speed guarantee.',
         'Recorded environments differ or are incomplete; times are shown without a speed leader.'),
    ]
    if any(_mapping(record.get('test')) for _, record in runs):
        definitions.append(('heldout_r2', 'Held-out forecast R²', 'max', '_heldout_basis',
                            'Higher is better on the same recorded held-out interval; the worst variable is shown.',
                            'Held-out splits or forecast protocols differ or are incomplete; no forecast leader is established.'))
    return {'verified_input': verified, 'runs': rows,
            'metrics': [_metric(rows, verified, *definition) for definition in definitions],
            'notes': notes}
