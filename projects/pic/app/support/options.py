"""The collapsed "More options" block of the search page.

Defaults are the stored settings of the record, so a search can be started without
opening it. Only values that differ from those defaults become overrides.
"""

from dataclasses import asdict

import streamlit as st
import yaml

from .common import search_config, variant_label, variants

DERIVATIVES = {'FD': 'finite differences', 'poly': 'polynomial fits (noisy data)', 'spectral': 'spectral'}


def _effort(pop, epochs):
    return {'Quick': (max(4, pop // 2), max(2, epochs // 2)),
            'Standard': (pop, epochs),
            'Thorough': (pop * 2, epochs * 2)}


def search_options(name):
    """Render the options for ``name``; return ``(variant, seed, overrides, problem)``.

    ``problem`` is None when the record cannot be loaded (the error is shown here)."""
    with st.expander('More options'):
        basic, expert = st.tabs(['Basic', 'Expert'])
        with basic:
            c1, c2 = st.columns([3, 1])
            variant = c1.selectbox('method', variants(), format_func=variant_label, key=f'variant_{name}')
            seed = int(c2.number_input('seed', 0, 999, 0, key=f'seed_{name}'))
        try:
            cfg, problem, resolved = search_config(name, variant)
        except (FileNotFoundError, ValueError, OSError) as exc:
            st.error(f'This record is unavailable: {exc}')
            return variant, seed, {}, None
        if cfg.get('method') != 'epde':
            with basic:
                st.caption('This method is sparse regression outside EPDE; the search options do not apply.')
            return variant, seed, {}, problem

        key = f'{name}_{variant}'
        prep, space, evo = resolved['preprocessing'], resolved['search_space'], resolved['evolution']
        changes = {}
        with basic:
            now = prep.get('default_preprocessor_type', 'FD')
            choices = list(DERIVATIVES) if now in DERIVATIVES else list(DERIVATIVES) + [now]
            method = st.radio('derivatives', choices, index=choices.index(now), horizontal=True,
                              format_func=lambda m: DERIVATIVES.get(m, m), key=f'prep_{key}')
            if method != now:
                changes.setdefault('preprocessing', {})['default_preprocessor_type'] = method
            pop, epochs = int(evo.get('population_size', 16)), int(evo.get('training_epochs', 5))
            presets = _effort(pop, epochs)
            effort = st.radio('search effort', list(presets), index=1, horizontal=True, key=f'effort_{key}',
                              format_func=lambda e: f'{e} ({presets[e][0]} candidates × {presets[e][1]} epochs)')
            a, b = st.columns(2)
            terms_now = int(space.get('equation_terms_max_number', 6))
            terms = a.slider('largest number of terms', 2, 15, terms_now, key=f'terms_{key}')
            fmax = space.get('equation_factors_max_number')
            two = isinstance(fmax, dict) and max(fmax.get('factors_num', [1])) >= 2
            products = b.toggle('allow products such as u·uₓ', value=two, key=f'prod_{key}')
            if terms != terms_now:
                changes.setdefault('search_space', {})['equation_terms_max_number'] = terms
            if products != two:
                changes.setdefault('search_space', {})['equation_factors_max_number'] = (
                    {'factors_num': [1, 2], 'probas': [0.65, 0.35]} if products else 1)

        with expert:
            from epde.interface.search_config import METRIC_MENUS, SPARSITY_REGISTRY, load_search_config
            current = asdict(load_search_config(resolved).objectives)
            a, b = st.columns(2)
            picks = {}
            for column, field, label in [(a, 'discrepancy_metric', 'discrepancy'),
                                         (b, 'second_objective', 'second objective'),
                                         (a, 'instability_metric', 'instability estimator')]:
                menu = METRIC_MENUS[field]
                picks[field] = column.selectbox(label, menu, index=menu.index(current[field]),
                                                key=f'{field}_{key}')
            sparsity_now = resolved.get('objectives', {}).get('sparsity_cls', 'vwsr')
            sparsity_now = sparsity_now if sparsity_now in SPARSITY_REGISTRY else 'vwsr'
            picks['sparsity_cls'] = b.selectbox('sparsity', SPARSITY_REGISTRY,
                                                index=SPARSITY_REGISTRY.index(sparsity_now), key=f'sparsity_{key}')
            current['sparsity_cls'] = sparsity_now
            for field, value in picks.items():
                if value != current[field]:
                    changes.setdefault('objectives', {})[field] = value

            dpow_now = int(space.get('data_fun_pow', 1))
            dpow = a.slider('highest power of a variable', 1, 4, dpow_now, key=f'dpow_{key}')
            if dpow != dpow_now:
                changes.setdefault('search_space', {})['data_fun_pow'] = dpow
            orders = prep['max_deriv_order'] if isinstance(prep['max_deriv_order'], list) else [prep['max_deriv_order']]
            new_orders = [int(col.number_input(f'highest derivative in {axis}', 1, 6, int(k), key=f'ord_{axis}_{key}'))
                          for col, axis, k in zip(st.columns(len(orders)), problem.axis_names, orders)]
            if new_orders != list(orders):
                changes.setdefault('preprocessing', {})['max_deriv_order'] = new_orders
            exact = st.toggle('set candidates and epochs exactly', key=f'exact_{key}')
            if exact:
                a, b = st.columns(2)
                pop_e = a.slider('candidates', 4, 96, presets[effort][0], step=4, key=f'pop_{key}')
                epochs_e = b.slider('epochs', 1, 50, presets[effort][1], key=f'epochs_{key}')
            else:
                pop_e, epochs_e = presets[effort]
            if pop_e != pop:
                changes.setdefault('evolution', {})['population_size'] = pop_e
            if epochs_e != epochs:
                changes.setdefault('evolution', {})['training_epochs'] = epochs_e
            overrides = {'search': changes} if changes else {}
            st.caption('Changes from the stored settings of this record:')
            st.code(yaml.safe_dump(overrides, sort_keys=False) if overrides else '# none', language='yaml')
    return variant, seed, overrides, problem
