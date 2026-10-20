"""Independent native/fixed controls for DeepMoD and DISCOVER."""
import streamlit as st

from epde_bench import pdecomp, pdecomp_methods as methods
from support.discovery_lab.pysindy_method import shape_of_arrays


@st.cache_data(show_spinner=False)
def starting_settings(method, dataset, protocol, kind, orders):
    return methods.describe(method, dataset, protocol=protocol, kind=kind, orders=orders)


def render_settings(method, record_name=None, problem=None, arrays=None, source=None, heavy=False):
    label = methods.LABELS[method]
    prefix = f'lab_{method}_{record_name or "data"}_'
    st.markdown(f'**{label} · PDEComp**')
    protocol = st.pills('Protocol', ['native', 'fixed'], default='native',
                        selection_mode='single', key=prefix + 'protocol') or 'native'
    if method == 'deepmod':
        st.caption('Native: DeePyMoD neural network, automatic differentiation, Library1D, '
                   'sparsity estimation and least-squares constraint. Fixed: sparse estimation '
                   'on PDEComp’s shared derivatives and candidate library.')
    else:
        st.caption('Native: DISCOVER recurrent controller searches its symbolic operators; '
                   'no shared feature matrix is supplied. Fixed: DISCOVER searches shared library terms.')
    if problem is not None:
        kind, count = problem.kind, len(problem.variables)
        orders = pdecomp.target_orders(problem)
    elif record_name:
        from epde_bench.datasets import REGISTRY
        from support.discovery_lab.catalog import declared_equations
        kind = REGISTRY[record_name].kind
        count = len(declared_equations(record_name)) or 1
        orders = None
    else:
        kind, count = shape_of_arrays(arrays)
        variables = [k[4:] for k in arrays if k.startswith('var_')]
        order = st.pills('Target time derivative', [1, 2], default=1,
                         selection_mode='single', key=prefix + 'target') or 1
        orders = {v: order for v in variables}
    reason = methods.unsupported_reason(method, kind, count, protocol, orders)
    if reason:
        st.warning(reason + '. Choose another protocol or deselect this method in step 1.')
        return None
    if kind == 'pde_2d' and not pdecomp.supported(record_name):
        st.warning('DeepMoD fixed 2D support requires the PDEComp Navier–Stokes record.')
        return None
    base = starting_settings(method, record_name, protocol, kind, orders)
    current = base.get('direct_optimizer', {}) if method == 'deepmod' and protocol == 'fixed' else base
    shown = dict(current)
    if method == 'discover' or protocol == 'native':
        budget_keys = ('max_iterations', 'max_samples') if method == 'deepmod' else ('n_samples', 'batch_size')
        caps = {'max_iterations': 250, 'max_samples': 1000} if method == 'deepmod' else {'n_samples': 1000, 'batch_size': 100}
        full_budget = {'max_iterations': 5000, 'max_samples': 10000} if method == 'deepmod' else {'n_samples': 5000, 'batch_size': 250}

        def reset_budget():
            budget_key = prefix + protocol + '_budget'
            if st.session_state.get(budget_key) is None:
                st.session_state[budget_key] = 'PDEComp profile' if heavy else 'Quick demo'
            saved = st.session_state.get('lab_widgets_saved', {})
            for name in budget_keys:
                widget_key = prefix + protocol + '_' + name
                value = int(current.get(name, full_budget[name]))
                if st.session_state[budget_key] == 'Quick demo':
                    value = min(value, caps[name])
                st.session_state[widget_key] = value
                saved[widget_key] = value

        budget = st.segmented_control('Run budget', ['Quick demo', 'PDEComp profile'],
                                      default='PDEComp profile' if heavy else 'Quick demo',
                                      selection_mode='single', key=prefix + protocol + '_budget',
                                      on_change=reset_budget)
        if budget == 'Quick demo':
            shown.update({k: min(int(current.get(k, limit)), limit) for k, limit in caps.items()})
            st.caption('Quick demo uses a smaller training budget. Full PDEComp settings are '
                       'available above; short training does not guarantee equation recovery.')
        else:
            st.caption('The full PDEComp training budget can take several minutes. '
                       'Hosted searches have a time limit; use Quick demo for a short run.')
    changes = {}
    if method == 'deepmod' and protocol == 'fixed':
        value = st.selectbox('Sparse estimator', ['pdefind', 'threshold'],
                             index=['pdefind', 'threshold'].index(current.get('type', 'threshold')),
                             key=prefix + protocol + '_type')
        if value != current.get('type'):
            changes['type'] = value
        controls = (('threshold', 'Sparsity threshold', 0.0, 1e4, 0.1),
                    ('pdefind_lam', 'PDEFIND ridge penalty', 0.0, 1e4, 1e-3),
                    ('pdefind_dtol', 'PDEFIND threshold step', 1e-12, 1e4, 0.1),
                    ('coefficient_tol', 'Drop coefficients below', 0.0, 1e4, 0.0))
    elif method == 'deepmod':
        controls = (('max_iterations', 'Training iterations', 1, 10000, 5000),
                    ('max_samples', 'Maximum training samples', 100, 10000, 10000),
                    ('learning_rate', 'Learning rate', 1e-6, 0.1, 0.002),
                    ('threshold', 'Sparsity threshold', 0.0, 1.0, 0.1),
                    ('poly_order', 'Polynomial order', 1, 4, 3),
                    ('diff_order', 'Spatial derivative order', 1, 4, 4))
    else:
        controls = (('n_samples', 'Expression samples', 10, 50000, 5000),
                    ('batch_size', 'Controller batch size', 10, 500, 250),
                    ('epsilon', 'Risk-seeking fraction', 0.001, 1.0, 0.05),
                    ('max_length', 'Maximum expression length', 1, 64, 15),
                    ('max_add_count', 'Maximum additions', 1, 15, 8))
    cols = st.columns(2)
    for i, (key, title, low, high, default) in enumerate(controls):
        cast = int if type(low) is int else float
        initial = cast(shown.get(key, default))
        value = cols[i % 2].number_input(title, cast(low), cast(high), initial,
                                         key=prefix + protocol + '_' + key,
                                         **({'format': '%g', 'step': 0.0001 if key == 'learning_rate' else 0.01}
                                            if cast is float else {}))
        if value != cast(current.get(key, default)):
            changes[key] = value
    methods.validate_overrides(method, changes, protocol)
    if method == 'discover' and changes.get('n_samples', base['n_samples']) < changes.get('batch_size', base['batch_size']):
        st.error('Expression samples must be at least the batch size.')
        return None
    st.caption('Starting settings: ' + ('PDEComp’s dataset profile.' if pdecomp.supported(record_name)
                                       else 'PDEComp’s defaults for new data.'))
    st.caption(f'Source: [{pdecomp.PDECOMP_REPO}@{pdecomp.PDECOMP_REV[:7]}]'
               f'(https://github.com/{pdecomp.PDECOMP_REPO}/tree/{pdecomp.PDECOMP_REV}/{method})')
    settings = {'protocol': protocol, 'overrides': changes, 'target_orders': orders or {}}
    if heavy:
        return {'method': method, 'variant': f'{method}_pdecomp_{protocol}',
                'overrides': {method: {'overrides': changes}}}
    return dict(settings, kind=kind)


def review(method, dataset, settings, seed=0):
    return {'method': methods.LABELS[method], 'protocol': settings['protocol'],
            'source': f'{pdecomp.PDECOMP_REPO}@{pdecomp.PDECOMP_REV}',
            'target_orders': settings.get('target_orders') or 'from the known law',
            'changed': settings['overrides'],
            'effective': methods.describe(method, dataset, settings['overrides'], settings['protocol'],
                                          kind=settings.get('kind', 'ode'), orders=settings.get('target_orders'), seed=seed)}
