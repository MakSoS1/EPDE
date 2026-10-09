import streamlit as st

from support import diagrams
from support.common import link as page_link, page

page('How EPDE works', '🧭')
st.caption('Choose a stage to see what it does and where to try it in the app.')

STAGES = {
    'settings': ('Search settings',
        'Grouped settings (domain, preprocessing, search space, objectives, solver, evolution, '
        'runtime) with documented defaults; a file or dictionary overrides them, keyword '
        'arguments override the file. The resolved settings are published for all operators.',
        'views/5_Run_search.py'),
    'domain': ('Domain',
        'The grid of one experiment, one array per axis with time first, and the boundary '
        'excluded from fitting.',
        'views/1_Data_sets.py'),
    'trajectory': ('Data and derivatives',
        'Attaches the data; the preprocessing pipeline smooths the fields and computes '
        'derivatives (finite differences, polynomial fits, spectral, neural network). Everything '
        'is stored in the tensor cache.',
        'views/3_Derivatives.py'),
    'pool': ('Token pool',
        'Families of tokens: the variables and their derivatives with powers, plus families given'
        ' by the user (coordinates, sine and cosine, stored fields, custom functions). New terms '
        'are sampled from the pool.',
        'views/5_Run_search.py'),
    'population': ('Initial population',
        'Random systems of equations within the size limits: each equation is a set of terms, '
        'each term a product of tokens.',
        None),
    'fit': ('Fit of a candidate',
        'Every term of an equation is tried as the left-hand side. For each choice the sparsity '
        'operator decides which terms stay, coefficients are computed by least squares, and the '
        'candidate is scored; the best split is kept.',
        'views/2_Signal_check.py'),
    'sparsity': ('Sparsity',
        'Removes terms that do not carry weight. The default derives the penalty of each term '
        'from how much its coefficient varies across windows of the data.',
        'views/5_Run_search.py'),
    'objectives': ('Objectives',
        'Two values per candidate: how well the two sides agree, and either how stable the '
        'coefficients are across parts of the domain or how large the equation is.',
        'views/5_Run_search.py'),
    'evolution': ('Evolution',
        'Each epoch: parents are chosen from neighbouring subproblems, crossover exchanges '
        'equations and terms, mutation replaces terms, offspring are fitted and scored and '
        'inserted into the Pareto levels.',
        'views/5_Run_search.py'),
    'front': ('Pareto front',
        'The first non-dominated level: equations that trade accuracy against stability or size. '
        'One of them is chosen as the answer, e.g. the compromise point.',
        'views/5_Run_search.py'),
    'solve': ('Solve and check',
        'Optional: a discovered system is solved (built-in solver, DeepXDE, bases) and compared '
        'with the data.',
        'views/7_Results.py'),
}
EDGES = [('settings', 'domain'), ('settings', 'trajectory'), ('domain', 'trajectory'), ('trajectory', 'pool'),
         ('settings', 'pool'), ('pool', 'population'), ('population', 'fit'), ('fit', 'sparsity'),
         ('sparsity', 'objectives'), ('objectives', 'evolution'), ('evolution', 'fit'),
         ('evolution', 'front'), ('front', 'solve')]

chosen = st.radio('stage', list(STAGES), format_func=lambda k: STAGES[k][0], horizontal=True)


def graph(selected):
    lines = ['digraph G {', 'rankdir=TB; nodesep=0.4; ranksep=0.35;', 'node [shape=box, style="rounded,filled", fontname="Helvetica", '
             'fillcolor="#eef2f8", color="#9aa8bd"];', 'edge [color="#8a94a6"];']
    for key, (title, *_rest) in STAGES.items():
        fill = '#4c78a8' if key == selected else '#eef2f8'
        font = 'white' if key == selected else 'black'
        lines.append(f'{key} [label="{title}", fillcolor="{fill}", fontcolor="{font}"];')
    for a, b in EDGES:
        style = ' [style=dashed, label="next epoch"]' if (a, b) == ('evolution', 'fit') else ''
        lines.append(f'{a} -> {b}{style};')
    lines.append('}')
    return '\n'.join(lines)


st.graphviz_chart(graph(chosen))
title, text, link = STAGES[chosen]
st.subheader(title)
st.write(text)
if link:
    page_link(link, label='Try it in the app', icon='👉')

from support.interfaces import stage_interfaces
stage_interfaces(chosen)

# Each stage opens the matching view of the architecture model.
STAGE_VIEWS = {
    'settings': ('stageSettings', False), 'domain': ('stageData', False), 'trajectory': ('stageData', False),
    'pool': ('stagePool', False), 'population': ('stagePopulation', False), 'fit': ('stageFit', False),
    'sparsity': ('stageSparsity', False), 'objectives': ('stageObjectives', False),
    'evolution': ('stageEvolution', False), 'front': ('stageFront', False), 'solve': ('stageSolve', False),
}
MODEL_VIEWS = {
    'Landscape: EPDE, the data-set layer and external tools': ('index', False),
    'Packages of EPDE': ('epdeOverview', False),
    'One search from data to equations (sequence)': ('workflow', True),
    'One epoch of the evolution (sequence)': ('epoch', True),
    'Fit and score of one candidate (sequence)': ('candidate', True),
    'Evolutionary operators': ('operatorsView', False),
    'Data-set and benchmark layer': ('picView', False),
    'One benchmark run (sequence)': ('benchmarkRun', True),
}
if diagrams.available():
    view, sequence = STAGE_VIEWS[chosen]
    st.markdown('**Inside this stage** · click the diagram to explore the whole model')
    diagrams.likec4_view(view, height=380, sequence=sequence)
    with st.expander('Explore the whole architecture'):
        title = st.selectbox('view', list(MODEL_VIEWS), key='model_view')
        diagrams.likec4_view(*MODEL_VIEWS[title][:1], height=600, sequence=MODEL_VIEWS[title][1])

with st.expander('Watch a short evolution', expanded=False):
    st.markdown('Run a small example to observe the actual candidate front after each epoch. '
                'Four candidates evolve for three epochs on 121 exact samples of a damped '
                'oscillator. Exact derivatives isolate the evolutionary step from '
                'differentiation. The existing objectives and operators are used with a '
                'smaller population and equation size for this lesson.')
    st.latex(r'u(t)=e^{-t/4}\sin t,\qquad u_{tt}+0.5u_t+1.0625u=0')
    if st.button('Run short evolution', key='run_optimizer_demo'):
        from support.optimizer_process import run_optimizer_subprocess
        try:
            with st.spinner('Evolving a small population…'):
                st.session_state['optimizer_demo_result'] = run_optimizer_subprocess()
        except Exception as exc:
            st.session_state.pop('optimizer_demo_result', None)
            st.error(f'The short evolution could not complete: {exc}')
    result = st.session_state.get('optimizer_demo_result')
    if result:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from support.common import show_figure
        from support.equations import equation_latex

        st.caption(f"Observed live run: {result['seconds']:.3f} seconds. "
                   'A front may remain unchanged; improvement is not guaranteed at every epoch.')
        st.dataframe(pd.DataFrame(result['rows']), hide_index=True, width='stretch')
        st.caption('The smallest values in the two columns may belong to different candidates.')
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.3))
        for entry in result['history']:
            values = np.asarray([solution['obj_fun'] for solution in entry['front']], dtype=float)
            axes[0].scatter(values[:, 0], values[:, 1], s=45 + 20 * entry['epoch'],
                            alpha=0.6, label=f"Epoch {entry['epoch']}")
        axes[0].set(xlabel='Discrepancy', ylabel='Coefficient instability',
                    title='Observed fronts after each epoch')
        axes[0].legend()
        axes[1].plot([row['Epoch'] for row in result['rows']],
                     [row['Front size'] for row in result['rows']], 'o-')
        axes[1].set(xlabel='Epoch', ylabel='Number of candidates', title='Front size')
        axes[1].set_xticks([row['Epoch'] for row in result['rows']])
        fig.tight_layout()
        show_figure(fig)
        st.markdown('**Final discovered equations**')
        for equation in result['equations']:
            st.latex(equation_latex(equation, axes=('t',)))
        st.caption('Compare these discovered terms with the known oscillator law above. '
                   'This short example demonstrates the process, rather than reliability '
                   'across data sets or noisy measurements.')
