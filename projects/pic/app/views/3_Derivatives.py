import numpy as np
import streamlit as st

from support.common import dataset_label, dataset_names, page, search_config, show_figure
from support.equations import factor_latex

page('Derivatives', '📈')
st.caption('EPDE builds candidate equations from derivatives that it computes from the data. Noise in the '
           'data is amplified by differentiation, and a distorted derivative hides the terms that contain '
           'it. This page adds noise to a record, differentiates it with each method, and measures how far '
           'the result is from the derivative of the clean data.')

PREPROCESSORS = {'FD': 'finite differences', 'poly': 'polynomial fits', 'spectral': 'spectral'}
names = [n for n in dataset_names(kinds=('ode', 'ode_system', 'pde_1d')) if n not in ('dp_video',)]
c1, c2, c3 = st.columns([3, 2, 3])
name = c1.selectbox('record', names, format_func=dataset_label)
noise = c2.slider('noise, % of the standard deviation', 0.0, 10.0, 2.0, 0.5)
chosen = c3.multiselect('methods to compare', list(PREPROCESSORS), default=['FD', 'poly'],
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


def derivative_latex(i):
    axis, order = columns[i]
    name = f'd^{order}{var}/dx{axis}' if order > 1 else f'd{var}/dx{axis}'
    return factor_latex(name, {}, problem.axis_names)


col = c5.selectbox('derivative', range(len(columns)),
                   format_func=lambda i: f"order {columns[i][1]} along {problem.axis_names[columns[i][0]]}")
st.markdown(f'Compared derivative: ${derivative_latex(col)}$. Reference: finite differences of the '
            '**clean** data. Each method below works on the **noisy** data, as a search would.')

if st.button('Compute', type='primary') and chosen:
    import copy
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
    st.session_state['derivatives'] = ((name, noise, var, col), results,
                                       clean[var][:, col].reshape(problem.shape), noisy[var])


def judge(err):
    if err < 0.05:
        return 'reliable'
    if err < 0.3:
        return 'noticeably distorted'
    return 'unreliable: terms with this derivative will be hard to find'


state = st.session_state.get('derivatives')
if state and state[0] == (name, noise, var, col):
    import matplotlib.pyplot as plt
    _, results, reference, noisy_field = state
    if not results:
        st.info('No method produced valid derivatives. Adjust the selection and try again.')
        st.stop()
    inner = tuple(slice(max(1, n // 10), n - max(1, n // 10)) for n in problem.shape)
    scale = float(np.std(reference[inner])) or 1.0
    errors = {prep: float(np.sqrt(np.mean((values[inner] - reference[inner]) ** 2))) / scale
              for prep, values in results.items()}

    st.subheader('Error of each method')
    st.caption('Root-mean-square difference from the reference derivative, divided by its standard deviation, '
               'away from the edges (10 % of each axis is left out). 0 % is a perfect derivative; 100 % means '
               'the error is as large as the derivative itself.')
    cols = st.columns(len(errors))
    for c, (prep, err) in zip(cols, errors.items()):
        c.metric(PREPROCESSORS[prep], f'{err:.1%}')
        c.caption(judge(err))
    best = min(errors, key=errors.get)
    st.info(f"At {noise:g} % noise the best method for ${derivative_latex(col)}$ is "
            f"**{PREPROCESSORS[best]}** ({errors[best]:.1%}). Noise added to the data: {noise:g} % of its "
            'standard deviation; differentiation turned it into the errors above.')

    st.subheader('Data and derivatives')
    if problem.dim == 0:
        t = np.asarray(problem.grids[0]).ravel()
        fig, axes = plt.subplots(len(results) + 1, 1, figsize=(10, 2.2 * (len(results) + 1)),
                                 squeeze=False, sharex=True)
        axes[0, 0].plot(t, noisy_field, color='0.6', lw=.8, label='noisy data')
        axes[0, 0].plot(t, problem.data[var], 'k', lw=1.0, label='clean data')
        axes[0, 0].set_ylabel(f'${var}$')
        axes[0, 0].legend(fontsize=8)
        for ax, (prep, values) in zip(axes[1:, 0], results.items()):
            ax.plot(t, values, lw=.8, color='C3', label=f'{PREPROCESSORS[prep]} on noisy data')
            ax.plot(t, reference, 'k', lw=1.2, label='reference (clean data)')
            span = np.ptp(reference) or 1.0
            ax.set_ylim(reference.min() - .25 * span, reference.max() + .25 * span)
            ax.set_ylabel(f'${derivative_latex(col)}$')
            ax.legend(fontsize=8)
        axes[-1, 0].set_xlabel(problem.axis_names[0])
    else:
        x = np.unique(np.asarray(problem.grids[1]))
        k = st.slider('time index', 0, problem.shape[0] - 1, problem.shape[0] // 2)
        fig, (a, b) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
        a.plot(x, noisy_field[k], color='0.6', lw=.8, label='noisy data')
        a.plot(x, problem.data[var][k], 'k', lw=1.0, label='clean data')
        a.set_ylabel(f'${var}$')
        a.legend(fontsize=8)
        for prep, values in results.items():
            b.plot(x, values[k], lw=.8, label=f'{PREPROCESSORS[prep]} on noisy data')
        b.plot(x, reference[k], 'k', lw=1.2, label='reference (clean data)')
        span = np.ptp(reference[k]) or 1.0
        b.set_ylim(reference[k].min() - .25 * span, reference[k].max() + .25 * span)
        b.set_ylabel(f'${derivative_latex(col)}$')
        b.set_xlabel(problem.axis_names[1])
        b.legend(fontsize=8)
    fig.tight_layout()
    show_figure(fig)
    st.caption('Next step: the Signal check page shows whether the known law is still visible with these '
               'derivatives; Run a search uses the method chosen under More options.')
