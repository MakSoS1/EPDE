import io
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yaml

from support import jobs
from support.common import APP_RESULTS, page, show_figure
from support.custom_run import validate_arrays

page('Your data', '📤')
st.caption('Upload a file and press Start. Accepted: a CSV table with a time column and one column per '
           'variable; a CSV or NPY matrix of a field (time along rows, space along columns); an NPZ archive '
           'with coordinates (see About this page).')

upload = st.file_uploader('file', type=['csv', 'npy', 'npz'], label_visibility='collapsed')
if upload is None:
    st.stop()


def _csv_has_header(raw):
    first = raw.decode('utf-8-sig').splitlines()[0]
    try:
        [float(cell) for cell in first.split(',')]
        return False
    except ValueError:
        return True


arrays = {}
try:
    raw, suffix = upload.getvalue(), Path(upload.name).suffix.lower()
    if suffix == '.npz':
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            arrays = {k: archive[k] for k in archive.files}
        if len([k for k in arrays if k.startswith('axis_') and k != 'axis_names']) > 2:
            raise ValueError('This page supports time series and fields with one space axis.')
    elif suffix == '.csv' and _csv_has_header(raw):
        df = pd.read_csv(io.BytesIO(raw))
        if len(df.columns) < 2:
            raise ValueError('The table needs a time column and at least one variable column.')
        time_col = df.columns[0]
        arrays['axis_0'] = df[time_col].to_numpy(float)
        arrays['axis_names'] = np.array(['t'])
        for column in df.columns[1:]:
            arrays[f'var_{column}'] = df[column].to_numpy(float)
        st.caption(f'Time series: time in “{time_col}”, variables {", ".join(map(str, df.columns[1:]))}.')
    else:
        field = (np.load(io.BytesIO(raw), allow_pickle=False) if suffix == '.npy'
                 else np.loadtxt(io.StringIO(raw.decode('utf-8-sig')), delimiter=','))
        if field.ndim != 2:
            raise ValueError(f'Expected a matrix (time × space), got shape {field.shape}.')
        c1, c2, c3, c4 = st.columns(4)
        t0, t1 = c1.number_input('t from', value=0.0), c2.number_input('t to', value=1.0)
        x0, x1 = c3.number_input('x from', value=0.0), c4.number_input('x to', value=1.0)
        arrays = {'axis_0': np.linspace(t0, t1, field.shape[0]), 'axis_1': np.linspace(x0, x1, field.shape[1]),
                  'axis_names': np.array(['t', 'x']), 'var_u': field}
        st.caption(f'Field u with {field.shape[0]} time steps and {field.shape[1]} points in space.')
    axes, axis_names, fields = validate_arrays(arrays)
except (ValueError, TypeError, OSError, UnicodeError, KeyError, IndexError) as exc:
    st.error(f'Cannot use this file: {exc}')
    st.stop()

import matplotlib.pyplot as plt  # noqa: E402

if len(axes) == 1:
    fig, ax = plt.subplots(figsize=(10, 2.4))
    for k, v in fields.items():
        ax.plot(axes[0], v, label=k)
    ax.set_xlabel(axis_names[0])
    ax.legend()
else:
    k = next(iter(fields))
    fig, ax = plt.subplots(figsize=(8, 3))
    im = ax.imshow(fields[k].T, origin='lower', aspect='auto', cmap='RdBu_r',
                   extent=[axes[0][0], axes[0][-1], axes[1][0], axes[1][-1]])
    ax.set_xlabel(axis_names[0])
    ax.set_ylabel(axis_names[1])
    ax.set_title(k)
    fig.colorbar(im, ax=ax)
show_figure(fig)
if not np.allclose(np.diff(axes[0]), np.diff(axes[0]).mean(), rtol=1e-3):
    st.warning('The time step is not uniform; derivatives use the supplied coordinates.')

dim = len(axes) - 1
with st.expander('More options'):
    a, b, c = st.columns(3)
    orders = [int(a.number_input('highest time derivative', 1, 3, 2))]
    if dim:
        orders.append(int(b.number_input('highest space derivative', 1, 4, 3)))
    prep = c.selectbox('derivatives', ['FD', 'poly'],
                       format_func={'FD': 'finite differences', 'poly': 'polynomial fits (noisy data)'}.get)
    a, b, c = st.columns(3)
    terms = a.slider('largest number of terms', 2, 15, 8)
    products = b.toggle('allow products such as u·uₓ', value=True)
    coords = c.toggle('allow the coordinates (t, t²)', value=False)
    a, b, c = st.columns(3)
    trig = a.number_input('sine/cosine of frequency (0 = none)', 0.0, 100.0, 0.0)
    effort = b.radio('search effort', ['Quick', 'Standard', 'Thorough'], index=1, horizontal=True)
    noise = c.number_input('add noise, % (experiments)', 0.0, 50.0, 0.0)

pop, epochs = {'Quick': (8, 3), 'Standard': (16, 5), 'Thorough': (32, 10)}[effort]
tokens = [{'family': 'grid', 'labels': ['x'], 'max_power': 2}] if coords else []
if trig:
    tokens.append({'family': 'fixed_trigonometric', 'freq': float(trig)})
settings = {'noise': float(noise), 'seed': 0, 'search': {
    'domain': {'boundary_width': 'auto'},
    'preprocessing': {'default_preprocessor_type': prep, 'preprocessor_kwargs': {}, 'max_deriv_order': orders},
    'search_space': {'data_fun_pow': 3, 'deriv_fun_pow': 1, 'equation_terms_max_number': terms,
                     'equation_factors_max_number': {'factors_num': [1, 2], 'probas': [0.65, 0.35]} if products else 1,
                     'tokens': tokens},
    'objectives': {}, 'solver': {'use_solver': False, 'device': 'cpu'},
    'evolution': {'population_size': pop, 'training_epochs': epochs},
    'runtime': {'verbose_params': {'show_iter_idx': False}, 'memory_for_cache': 2}}}

if st.button('Start', type='primary'):
    folder = APP_RESULTS / 'uploads'
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{Path(upload.name).stem}_{uuid.uuid4().hex[:6]}"
    np.savez(folder / f'{stem}.npz', **arrays)
    (folder / f'{stem}.yaml').write_text(yaml.safe_dump(settings, sort_keys=False), encoding='utf-8')
    job = jobs.start_custom(folder / f'{stem}.npz', folder / f'{stem}.yaml', f'uploaded: {upload.name}')
    st.session_state['run_job'] = job['id']
    st.switch_page('views/5_Run_search.py')
st.caption('The search opens on the Run a search page; earlier searches are listed on Results.')
