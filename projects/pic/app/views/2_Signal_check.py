import numpy as np
import pandas as pd
import streamlit as st

from support.common import dataset_label, dataset_names, load_problem, page, show_figure, variants
from support.compare import fitted_equation
from support.equations import equation_latex, system_latex, term_plain

page('Signal check', '🔍')
st.caption('No search happens here. The known law of the record is taken as given, its terms are computed '
           'from the (noisy) data, and their coefficients are fitted by least squares. If the law no '
           'longer explains the data, or one of its terms explains almost nothing, a search will '
           'struggle to find it.')

c1, c2, c3, c4 = st.columns([3, 2, 2, 1], vertical_alignment='bottom')
name = c1.selectbox('record', dataset_names(with_truth=True), format_func=dataset_label)
noise = c2.multiselect('noise, % of the standard deviation', [0, 0.5, 1, 2, 5, 10], default=[0, 1, 5])
variant = c3.selectbox('derivatives', [v for v in variants() if v in ('default', 'poly')],
                       format_func=lambda v: {'default': 'finite differences', 'poly': 'polynomial fits'}[v])
seed = c4.number_input('seed', 0, 999, 0)
if name == 'jhtdb_plane' and any(noise):
    st.warning('This record ships noise-free derivatives; only 0 % is allowed.')
    noise = [0]

problem = load_problem(name)
axes = problem.axis_names
st.markdown('**Known law of this record**')
st.latex(system_latex(problem.truth, axes))

if st.button('Run the check', type='primary') and noise:
    from epde_bench.truthcheck import check
    overrides = {'search': {'preprocessing': {'preprocessor_kwargs': {'mp_poolsize': 1}}}} if variant == 'poly' else None
    with st.spinner('evaluating terms…'):
        _, report = check(name, noise_levels=sorted(noise), seed=int(seed), variant=variant,
                          overrides=overrides, series=True)
    st.session_state['signal_check'] = ((name, variant, int(seed)), report)

WEAK = 1e-3          # R² lost when a term is dropped below which the term is practically invisible


def verdict(eq):
    target = term_plain(eq['equation'].split('=')[1], axes)
    weak = [term_plain(r['term'], axes) for r in eq['terms'] if r['r2_loss_if_dropped'] < WEAK]
    if eq['r2'] >= 0.99 and not weak:
        return 'success', f"The law explains {eq['r2']:.2%} of {target} and every term contributes: it is visible in the data."
    if eq['r2'] >= 0.9:
        text = f"The law still explains {eq['r2']:.1%} of {target}, but the fitted coefficients drift from the true ones."
        kind = 'warning'
    else:
        text = (f"The law explains only {eq['r2']:.1%} of {target}: the derivatives are dominated by noise, "
                'so a search on these data is unlikely to recover it.')
        kind = 'error'
    if weak:
        text += f" Barely visible: {', '.join(weak)} — dropping it costs less than {WEAK:.1%} of R²."
        kind = 'error' if kind == 'error' else 'warning'
    return kind, text


def plot_equation(eq, times):
    import matplotlib.pyplot as plt
    target, predicted = np.asarray(eq['target']), np.asarray(eq['predicted'])
    label = '$' + equation_latex(eq['equation'], axes).split('=')[0].strip() + '$'
    if times is not None and len(times) == len(target):
        fig, (a, b) = plt.subplots(2, 1, figsize=(10, 4.2), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        a.plot(times, target, color='0.55', lw=1.0, label=f'{label} computed from the data')
        a.plot(times, predicted, color='C3', lw=1.2, ls='--', label='right-hand side of the fitted law')
        a.legend(fontsize=8)
        b.plot(times, target - predicted, color='C0', lw=0.8)
        b.axhline(0, color='0.3', lw=0.5)
        b.set_ylabel('difference')
        b.set_xlabel(axes[0])
    else:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(target), min(4000, len(target)), replace=False)
        fig, a = plt.subplots(figsize=(5, 4.5))
        a.scatter(target[idx], predicted[idx], s=3, alpha=.4, color='C3')
        lo, hi = np.percentile(target, [0.5, 99.5])
        a.plot([lo, hi], [lo, hi], color='0.3', lw=0.8, label='perfect agreement')
        a.set_xlabel(f'{label} computed from the data')
        a.set_ylabel('right-hand side of the fitted law')
        a.legend(fontsize=8)
    fig.tight_layout()
    show_figure(fig)


state = st.session_state.get('signal_check')
if state and state[0] == (name, variant, int(seed)):
    _, report = state
    st.markdown('**Result for each noise level**')
    tabs = st.tabs([f"noise {block['noise']:g} %" for block in report])
    for tab, block in zip(tabs, report):
        with tab:
            for eq in block['equations']:
                kind, text = verdict(eq)
                getattr(st, kind)(text)
                left, right = st.columns(2)
                left.markdown('Known law')
                left.latex(equation_latex(eq['equation'], axes))
                right.markdown('Same terms, coefficients fitted to these data')
                right.latex(equation_latex(fitted_equation(eq), axes))
                plot_equation(eq, block.get('times'))
                st.dataframe(pd.DataFrame([{
                    'term': term_plain(r['term'], axes), 'true coefficient': r['true'],
                    'fitted coefficient': r['fitted'],
                    'share of R² lost without this term': r['r2_loss_if_dropped']} for r in eq['terms']]),
                    hide_index=True, width='stretch',
                    column_config={'share of R² lost without this term': st.column_config.NumberColumn(format='%.2e')})
    if len(report) > 1:
        st.markdown('**How the law fades with noise**')
        summary = pd.DataFrame([{'noise, %': block['noise'],
                                 'equation for': term_plain(eq['equation'].split('=')[1], axes),
                                 'R² of the known law': eq['r2']}
                                for block in report for eq in block['equations']])
        st.line_chart(summary.pivot_table(index='noise, %', columns='equation for', values='R² of the known law'))
        st.caption('R² = 1: the known law with fitted coefficients reproduces the target computed from the '
                   'data exactly. Values far below 1 mean the derivatives are too noisy for this law to show.')
