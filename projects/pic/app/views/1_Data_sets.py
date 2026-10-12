import json

import pandas as pd
import streamlit as st
import yaml

from support.common import KIND_NAMES, dataset_label, load_problem, page, registry_table, show_figure
from support.equations import system_latex

page('Data sets', '🗂️')

rows = pd.DataFrame(registry_table())
f1, f2, f3 = st.columns(3)
suites = f1.multiselect('suite', sorted(rows.suite.unique()), default=sorted(rows.suite.unique()))
classes = f2.multiselect('class', sorted(rows['class'].unique()), default=sorted(rows['class'].unique()))
law = f3.selectbox('law', ['any', 'known', 'unknown'])
view = rows[rows.suite.isin(suites) & rows['class'].isin(classes)]
if law != 'any':
    view = view[view['law known'] == (law == 'known')]
st.dataframe(view[['title', 'suite', 'class', 'source', 'law known']], hide_index=True,
             width='stretch')
st.caption('core: synthetic with a known law, the main benchmark; extended: slower or less standard; '
           'real: measurements; other: loads but is not benchmarked.')

if view.empty:
    st.stop()
name = st.selectbox('record', list(view.name), format_func=dataset_label)
loader_kwargs = {}
if name == 'sst':
    loader_kwargs['crop_ocean'] = st.toggle('largest ocean region finite on every day', value=True)
try:
    problem = load_problem(name, json.dumps(loader_kwargs, sort_keys=True))
except FileNotFoundError as exc:
    st.error(f'The data files of this record are missing: {exc}')
    st.stop()

left, right = st.columns([2, 3])
with left:
    st.subheader(problem.title)
    st.markdown(f"**class:** {KIND_NAMES.get(problem.kind, problem.kind)} · **source:** {problem.source}  \n"
                f"**axes:** {', '.join(problem.axis_names)} · **shape:** {problem.shape} · "
                f"**variables:** {', '.join(problem.variables)}")
    if problem.truth:
        st.markdown('**Known law**')
        st.latex(system_latex(problem.truth, problem.axis_names))
        for alt in problem.truth_alternatives:
            with st.expander('also accepted'):
                st.latex(system_latex(alt, problem.axis_names))
    else:
        st.markdown('**Known law:** not known')
    if problem.named_arrays:
        st.markdown('**Extra fields used as tokens:** ' + ', '.join(
            f"{k} ({'may stand alone' if v[1] else 'factor only'})" for k, v in problem.named_arrays.items()))
    if problem.derivs is not None:
        st.markdown('**Derivatives:** supplied with the data')
    st.caption(problem.notes)
with right:
    from epde_bench.plotting import plot_problem
    show_figure(plot_problem(problem))

with st.expander('configuration of this record'):
    from epde_bench.paths import CONFIG_DIR
    path = CONFIG_DIR / f'{name}.yaml'
    st.code(path.read_text(encoding='utf-8') if path.exists() else '# uses the shared protocol only', language='yaml')
with st.expander('summary'):
    st.code(problem.summary(), language=None)
