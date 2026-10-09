import pandas as pd
import streamlit as st

from support.common import dataset_label, dataset_names, load_problem, page, variants
from support.equations import term_plain

page('Signal check', '🔍')
st.caption('Is the known law still visible in the data? The terms of the law are fitted directly; '
           'a term that explains almost nothing will be hard to find.')

c1, c2, c3, c4 = st.columns([3, 2, 2, 1], vertical_alignment='bottom')
name = c1.selectbox('record', dataset_names(with_truth=True), format_func=dataset_label)
noise = c2.multiselect('noise, % of the standard deviation', [0, 0.5, 1, 2, 5, 10], default=[0, 1, 5])
variant = c3.selectbox('derivatives', [v for v in variants() if v in ('default', 'poly')],
                       format_func=lambda v: {'default': 'finite differences', 'poly': 'polynomial fits'}[v])
seed = c4.number_input('seed', 0, 999, 0)
if name == 'jhtdb_plane' and any(noise):
    st.warning('This record ships noise-free derivatives; only 0 % is allowed.')
    noise = [0]

if st.button('Run the check', type='primary') and noise:
    from epde_bench.truthcheck import check
    overrides = {'search': {'preprocessing': {'preprocessor_kwargs': {'mp_poolsize': 1}}}} if variant == 'poly' else None
    with st.spinner('evaluating terms…'):
        problem, report = check(name, noise_levels=sorted(noise), seed=int(seed), variant=variant,
                                overrides=overrides)
    st.session_state['signal_check'] = (name, report)

if st.session_state.get('signal_check', (None,))[0] == name:
    _, report = st.session_state['signal_check']
    axes = load_problem(name).axis_names
    summary = pd.DataFrame([{'noise, %': block['noise'], 'equation': term_plain(eq['equation'].split('=')[1], axes),
                             'R² of the true structure': eq['r2']}
                            for block in report for eq in block['equations']])
    st.subheader('R² of the true structure')
    st.line_chart(summary.pivot_table(index='noise, %', columns='equation', values='R² of the true structure'))
    st.dataframe(summary, hide_index=True, width='stretch')
    st.subheader('Contribution of each term')
    terms = pd.DataFrame([{'noise, %': block['noise'], 'equation': term_plain(eq['equation'].split('=')[1], axes),
                           'term': term_plain(row['term'], axes), 'true': row['true'], 'fitted': row['fitted'],
                           'R² lost if dropped': row['r2_loss_if_dropped']}
                          for block in report for eq in block['equations'] for row in eq['terms']])
    st.bar_chart(terms, x='term', y='R² lost if dropped', color='noise, %', stack=False)
    st.dataframe(terms, hide_index=True, width='stretch')
