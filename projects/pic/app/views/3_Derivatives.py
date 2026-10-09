import json

import numpy as np
import streamlit as st

from support.common import dataset_label, dataset_names, load_problem, page, search_config, show_figure

page('Derivatives', '📈')
st.caption('Derivatives computed from noisy data by different methods, compared with those from clean data.')

PREPROCESSORS = {'FD': 'finite differences', 'poly': 'polynomial fits', 'spectral': 'spectral'}
names = [n for n in dataset_names(kinds=('ode', 'ode_system', 'pde_1d')) if n not in ('dp_video',)]
c1, c2, c3 = st.columns([3, 2, 3])
name = c1.selectbox('record', names, format_func=dataset_label)
noise = c2.slider('noise, %', 0.0, 10.0, 2.0, 0.5)
chosen = c3.multiselect('preprocessors', list(PREPROCESSORS), default=['FD', 'poly'],
                        format_func=PREPROCESSORS.get)

cfg, problem, base = search_config(name)
if problem.derivs is not None:
    st.info('This record ships its derivatives; they are not recomputed.')
    st.stop()
c4, c5 = st.columns(2)
var = c4.selectbox('variable', problem.variables)
orders = base['preprocessing']['max_deriv_order']
orders = orders if isinstance(orders, list) else [orders] * len(problem.grids)
columns = [(axis, k) for axis, kmax in enumerate(orders) for k in range(1, int(kmax) + 1)]
col = c5.selectbox('derivative', range(len(columns)),
                   format_func=lambda i: f"order {columns[i][1]} along {problem.axis_names[columns[i][0]]}")

if st.button('Compute', type='primary') and chosen:
    import copy
    import matplotlib.pyplot as plt
    from epde_bench.preprocessing import prepare_data
    noisy = problem.noisy(noise, 0)
    clean_cfg = copy.deepcopy(base)
    clean_cfg['preprocessing']['default_preprocessor_type'] = 'FD'
    clean_cfg['preprocessing']['preprocessor_kwargs'] = {}
    with st.spinner('differentiating…'):
        _, clean = prepare_data(problem, problem.data, clean_cfg)
        results = {}
        for prep in chosen:
            c = copy.deepcopy(base)
            c['preprocessing']['default_preprocessor_type'] = prep
            c['preprocessing']['preprocessor_kwargs'] = {'mp_poolsize': 1} if prep == 'poly' else {}
            try:
                _, stacks = prepare_data(problem, noisy, c)
                results[prep] = stacks[var][:, col].reshape(problem.shape)
            except Exception as exc:              # noqa: BLE001
                st.warning(f'{PREPROCESSORS[prep]}: {exc}')
    st.session_state['derivatives'] = ((name, noise, var, col), results, clean[var][:, col].reshape(problem.shape))

state = st.session_state.get('derivatives')
if state and state[0] == (name, noise, var, col):
    import matplotlib.pyplot as plt
    _, results, reference = state
    inner = tuple(slice(max(1, n // 10), n - max(1, n // 10)) for n in problem.shape)
    scale = float(np.std(reference[inner])) or 1.0
    st.subheader('Relative error inside the domain')
    cols = st.columns(len(results) or 1)
    for c, (prep, values) in zip(cols, results.items()):
        err = float(np.sqrt(np.mean((values[inner] - reference[inner]) ** 2))) / scale
        c.metric(PREPROCESSORS[prep], f'{err:.1%}')
    if not results:
        st.info('No preprocessor produced valid derivatives. Adjust the selection and try again.')
        st.stop()
    if problem.dim == 0:
        t = np.asarray(problem.grids[0])
        fig, axes = plt.subplots(len(results), 1, figsize=(10, 2.4 * len(results)), squeeze=False, sharex=True)
        for ax, (prep, values) in zip(axes[:, 0], results.items()):
            ax.plot(t, values, lw=.8, label=f'{PREPROCESSORS[prep]}, noisy data')
            ax.plot(t, reference, 'k', lw=1.2, label='finite differences, clean data')
            ax.set_ylim(*np.percentile(reference, [0, 100]) * 1.5)
            ax.legend(fontsize=8)
        axes[-1, 0].set_xlabel(problem.axis_names[0])
    else:
        x = np.unique(np.asarray(problem.grids[1]))
        k = st.slider('time index', 0, problem.shape[0] - 1, problem.shape[0] // 2)
        fig, ax = plt.subplots(figsize=(10, 3.2))
        for prep, values in results.items():
            ax.plot(x, values[k], lw=.8, label=f'{PREPROCESSORS[prep]}, noisy data')
        ax.plot(x, reference[k], 'k', lw=1.2, label='finite differences, clean data')
        ax.set_xlabel(problem.axis_names[1])
        ax.legend(fontsize=8)
    fig.tight_layout()
    show_figure(fig)
