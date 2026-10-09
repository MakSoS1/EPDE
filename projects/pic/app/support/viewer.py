"""One view of a run record, used by the search, uploaded-data and results pages."""

import json

import numpy as np
import streamlit as st

from .common import load_problem, show_figure
from .equations import system_latex


def record_problem(record):
    if record.get('dataset') == 'custom':
        from .custom_run import problem_from_npz
        from epde_bench.identity import _file_hash
        from pathlib import Path
        expected = record.get('data_sha256') or (record.get('identity') or {}).get('data_sha256')
        if expected and _file_hash(Path(record['data_file'])) != expected:
            raise ValueError('Uploaded data changed since this run; reconstruction requires the original NPZ file.')
        return problem_from_npz(record['data_file'])
    loader = (record.get('config') or {}).get('loader', {})
    return load_problem(record['dataset'], json.dumps(loader, sort_keys=True))


def plot_pareto(objectives, selected=None, chosen=None):
    import matplotlib.pyplot as plt
    entries = [(i, o[:2]) for i, o in enumerate(objectives)
               if o is not None and len(o) >= 2 and np.all(np.isfinite(o[:2]))]
    pts = np.array([o for _, o in entries], dtype=float)
    positions = {original: local for local, (original, _) in enumerate(entries)}
    fig, ax = plt.subplots(figsize=(5, 3.6))
    if pts.size:
        ax.scatter(pts[:, 0], pts[:, 1], s=40, color='#4c78a8', zorder=2)
        for (original, _), (a, b) in zip(entries, pts):
            ax.annotate(str(original), (a, b), textcoords='offset points', xytext=(5, 4), fontsize=9)
        if selected in positions:
            ax.scatter(*pts[positions[selected]], s=160, facecolors='none', edgecolors='#e45756', lw=2,
                       label='compromise pick', zorder=3)
        if chosen in positions and chosen != selected:
            ax.scatter(*pts[positions[chosen]], s=160, facecolors='none', edgecolors='#54a24b', lw=2,
                       label='shown below', zorder=3)
        ax.legend(fontsize=8)
    ax.set_xlabel('discrepancy')
    ax.set_ylabel('second objective')
    ax.set_title('Final Pareto front', fontsize=10)
    ax.grid(alpha=.3)
    fig.tight_layout()
    return fig


def reconstruction_inputs(record, problem):
    """Reproduce the stored noise and differentiation protocol, including old baselines."""
    from epde_bench import load_config, resolve_for_problem
    data = problem.noisy(record.get('noise', 0), record.get('seed', 0))
    search = record.get('search_config')
    if search is None:
        cfg = record.get('config') or load_config(record['dataset'], record.get('variant', 'default'))
        search = resolve_for_problem(cfg, problem)
    from copy import deepcopy
    search = deepcopy(search)
    method = record.get('baseline', {}).get('differentiation', 'epde')
    if method != 'epde':
        search['_reconstruction_differentiation'] = method
    return data, search


def _verdict(m, truth):
    if not truth:
        return 'info', 'The governing law of these data is not known, so the result is not scored.'
    if m.get('success_selected'):
        text = 'The selected equation has exactly the terms of the known law.'
        if m.get('coef_error') is not None:
            text += f" Coefficient error {m['coef_error']:.2%}."
        return 'success', text
    if m.get('success_front'):
        return 'warning', ('The known law is among the candidates, but another equation was selected. '
                           'See All candidates.')
    return 'error', (f"The known law was not found (closest candidate differs by {m.get('hamming_min')} "
                     'terms).')


def _term_comparison(found, truth, axes):
    import pandas as pd
    from .compare import match_equations, term_rows
    from .equations import term_plain
    rows = []
    for truth_eq, found_eq in match_equations(found, truth):
        target = term_plain(truth_eq.split('=')[1], axes)
        for row in term_rows(found_eq, truth_eq, axes):
            rows.append({'equation for': target, **row} if len(truth) > 1 else row)
    if not rows:
        return
    st.markdown('**Term by term** (both written as *target = sum of terms*, scaled to the known target)')
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')


def render_record(record, key='rec'):
    status = record.get('status')
    if status != 'ok':
        st.error(record.get('reason') or record.get('error') or status)
        if record.get('traceback'):
            with st.expander('traceback'):
                st.code(record['traceback'])
        return

    problem = None
    try:
        problem = record_problem(record)
        axes = problem.axis_names
    except Exception as exc:                     # noqa: BLE001
        st.warning(f'data for this record could not be loaded: {exc}')
        axes = (record.get('problem') or {}).get('axis_names') or ('t', 'x', 'y', 'z')

    front = record.get('front') or []
    objectives = record.get('objectives') or []
    m = record.get('metrics') or {}
    selected = m.get('selected_index')
    truth = (record.get('problem') or {}).get('truth')
    if not front:
        st.warning('The search returned no equations.')
        return

    shown = selected if selected is not None and selected < len(front) else 0

    def _latex(system):
        try:
            st.latex(system_latex(system, axes))
        except Exception:                         # noqa: BLE001
            st.code('\n'.join(system))

    kind, text = _verdict(m, truth)
    if truth:
        left, right = st.columns(2)
        with left:
            st.markdown('**Known law** (what the search should recover)')
            _latex(truth)
        with right:
            st.markdown('**Found equation** (selected from the front)')
            _latex(front[shown])
        getattr(st, kind)(text)
        _term_comparison(front[shown], truth, axes)
    else:
        st.markdown('**Found equation**')
        _latex(front[shown])
        getattr(st, kind)(text)

    tab_data, tab_all, tab_details = st.tabs(['Compare with data', 'All candidates', 'Details'])
    with tab_all:
        groups = {}                               # identical systems are shown once
        for i, system in enumerate(front):
            groups.setdefault(tuple(system), []).append(i)
        left, right = st.columns([3, 2])
        with left:
            for system, members in groups.items():
                i = members[0]
                mark = ' · selected' if selected in members else ''
                same = f' ({len(members)} identical)' if len(members) > 1 else ''
                st.markdown(f'**[{i}]**{same}{mark}')
                try:
                    st.latex(system_latex(system, axes))
                except Exception:                 # noqa: BLE001
                    st.code('\n'.join(system))
        with right:
            show_figure(plot_pareto(objectives, selected, None))
            st.caption('Each point is a candidate: lower is better on both axes. '
                       'The circled one was selected without knowing the law.')

    with tab_data:
        if problem is None:
            st.info('The data of this record are not available, so the comparison cannot be drawn.')
        else:
            options = list(groups) if len(groups) > 1 else []
            chosen = shown
            if options:
                firsts = [members[0] for members in groups.values()]
                chosen = st.selectbox('candidate', firsts, index=firsts.index(groups[tuple(front[shown])][0]),
                                      format_func=lambda i: f'[{i}]' + (' selected' if i == shown else ''),
                                      key=f'{key}_chosen')
            st.caption('Black: computed from the data. Dashed: predicted by the equation; for ordinary '
                       'equations also its solution from the first point. Close agreement alone does not '
                       'prove the law.')
            try:
                from epde_bench.reconstruction import plot_reconstruction
                data, search = reconstruction_inputs(record, problem)
                fig, scores = plot_reconstruction(problem, front[chosen], search, data=data)
                show_figure(fig)
                if problem.dim == 1 and st.button('Also solve this field equation', key=f'{key}_pde_solve'):
                    _solve_field(problem, front[chosen], search, data)
            except Exception as exc:              # noqa: BLE001
                st.warning(f'comparison not available for this equation: {exc}')

    with tab_details:
        cols = st.columns(4)
        cols[0].metric('fit time, s', f"{record.get('fit_seconds', 0):.0f}")
        cols[1].metric('noise, %', f"{record.get('noise', 0):g}")
        cols[2].metric('seed', record.get('seed', 0))
        cols[3].metric('candidates', len(front))
        if truth:
            st.markdown(f"Truth on the front: **{m.get('success_front')}** (closest Hamming distance "
                        f"{m.get('hamming_min')}); selected equation correct: **{m.get('success_selected')}** "
                        f"(Hamming distance {m.get('hamming_selected')}).")
        st.markdown('**Equations in EPDE text form**')
        st.code('\n\n'.join('\n'.join(s) for s in front), language=None)
        st.markdown('**Settings of this run**')
        st.json(record.get('search_config') or record.get('config') or {}, expanded=False)
        st.download_button('download record (JSON)', json.dumps(record, indent=1, default=str),
                           file_name=f"{record.get('dataset')}_{record.get('variant')}_record.json",
                           mime='application/json', key=f'{key}_download')


def _solve_field(problem, system, search, data):
    """Method-of-lines solution of a 1-D field equation, shown next to the data."""
    from epde_bench.reconstruction import r2_score, simulate_pde
    st.caption('Explicit first-order evolution equations with space derivatives up to order two; '
               'the boundary values are taken from the data. This reconstructs the fitted record and '
               'is not an independent forecast.')
    result = simulate_pde(problem, system, search, data=data)
    if result is None:
        st.info('This equation form is not supported by the field integrator.')
        return
    if not result.success:
        st.error(f'Integration failed: {result.message}; coverage {result.coverage:.0%}.')
        return
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(problem.variables), 3, squeeze=False, figsize=(12, 3 * len(problem.variables)))
    for row, var in enumerate(problem.variables):
        observed, predicted = data[var], result.simulated[var]
        for ax, field, label in zip(axes[row], [observed, predicted, predicted - observed],
                                    ['data', 'solution of the equation', 'difference']):
            ax.imshow(field.T, origin='lower', aspect='auto')
            ax.set_title(f'{var}: {label} (R² = {r2_score(observed, predicted):.3f})' if label == 'solution of the equation'
                         else f'{var}: {label}')
            ax.set_xlabel('time sample')
            ax.set_ylabel('space sample')
    fig.tight_layout()
    show_figure(fig)
