"""Side-by-side comparison of a discovered (or fitted) equation with the known law.

Both equations are read in the form shown in the app, ``target = sum of terms``,
and scaled so that the known law's target has coefficient 1. Terms are matched
with the same canonical form as the benchmark metrics, so a row says whether a
term of the known law was found, missed, or added.
"""

import math
import re

from epde_bench.metrics import _equation_term_coefs

from .equations import term_plain

_COEF = re.compile(r'^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*\*\s*')


def _number(value):
    return f'{value:g}' if isinstance(value, float) else str(value)


def key_text(key):
    """EPDE text of a canonical term key; the empty key is the constant term."""
    if not key:
        return ''
    return ' * '.join(f"{name}{{{', '.join(f'{k}: {_number(v)}' for k, v in sorted(params))}}}"
                      for name, params in sorted(key, key=repr))


def term_label(key, axes):
    return term_plain(key_text(key), axes) if key else 'constant'


def _shown_coefs(eq_text, anchor=None):
    """Coefficients of ``target = Σ a_i t_i`` scaled to the anchor term, or ``None``."""
    parsed = _equation_term_coefs(eq_text)
    if parsed is None:
        return None, None
    coefs, target = parsed
    anchor = target if anchor is None else anchor
    scale = coefs.get(anchor, 0.0)
    if abs(scale) < 1e-12:
        return None, anchor
    return {k: -v / scale for k, v in coefs.items() if k != anchor and abs(v) > 0}, anchor


def match_equations(found, truth):
    """Pair every equation of the known law with the discovered equation that has the
    same target (or at least contains it); unmatched equations are paired with None."""
    pairs, used = [], set()
    for t_eq in truth:
        parsed = _equation_term_coefs(t_eq)
        anchor = parsed[1] if parsed else None
        best = None
        for i, f_eq in enumerate(found):
            if i in used:
                continue
            f_parsed = _equation_term_coefs(f_eq)
            if not f_parsed:
                continue
            if f_parsed[1] == anchor:
                best = i
                break
            if best is None and abs(f_parsed[0].get(anchor, 0.0)) > 1e-12:
                best = i
        if best is not None:
            used.add(best)
        pairs.append((t_eq, found[best] if best is not None else None))
    return pairs


def term_rows(found_eq, truth_eq, axes, found_label='found'):
    """Rows term / known / found / verdict for one pair of equations."""
    truth_coefs, anchor = _shown_coefs(truth_eq)
    found_coefs, _ = _shown_coefs(found_eq, anchor) if found_eq else (None, anchor)
    truth_coefs = truth_coefs or {}
    rows = []
    if found_eq and found_coefs is None:
        found_coefs = {}
        note = 'the found equation does not contain this target'
    else:
        note = None
    for key in list(truth_coefs) + [k for k in (found_coefs or {}) if k not in truth_coefs]:
        known = truth_coefs.get(key)
        got = (found_coefs or {}).get(key)
        if known is not None and got is not None:
            verdict = 'found'
            if abs(known) > 1e-12:
                verdict += f', coefficient off by {abs(got - known) / abs(known):.1%}'
        elif known is not None:
            verdict = note or 'missing'
        else:
            verdict = 'extra term'
        rows.append({'term': term_label(key, axes),
                     'known law': known, found_label: got, 'verdict': verdict})
    return rows


def fitted_equation(eq_report):
    """EPDE text of the known law with the coefficients fitted by the signal check."""
    lhs, rhs = eq_report['equation'].split('=', 1)
    chunks = [c for c in lhs.split('+') if c.strip()]
    parts = []
    for chunk, row in zip(chunks, eq_report['terms']):
        body = _COEF.sub('', chunk.strip())
        parts.append(f"{row['fitted']:.6g} * {body}")
    intercept = eq_report.get('intercept', 0.0)
    scale = eq_report.get('target_std') or 1.0
    if intercept and math.isfinite(intercept) and abs(intercept) > 1e-3 * scale:
        parts.append(f'{intercept:.6g}')
    return ' + '.join(parts) + ' = ' + rhs.strip()
