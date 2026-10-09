"""Paths, imports and cached loaders shared by all pages."""

import json
import os
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
PIC_DIR = APP_DIR.parent
REPO_ROOT = PIC_DIR.parent.parent
for path in (str(PIC_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from epde_bench import env  # noqa: E402  pins BLAS threads before numpy is imported

env.pin_blas_threads(1)

import streamlit as st  # noqa: E402

from epde_bench.paths import RESULTS_DIR  # noqa: E402

APP_RESULTS = RESULTS_DIR / 'app'
KIND_NAMES = {'ode': 'ODE', 'ode_system': 'ODE system', 'pde_1d': 'PDE, 1 space dimension',
              'pde_2d': 'PDE, 2 space dimensions', 'pde_3d': 'PDE, 3 space dimensions'}


def page(title, icon):
    st.set_page_config(page_title=f'{title} · EPDE', page_icon=icon, layout='wide')
    st.title(f'{icon} {title}')
    from .guide import reading_notes
    reading_notes(title)
    from .interfaces import page_interfaces
    page_interfaces(title)


@st.cache_data(show_spinner=False)
def registry_table():
    from epde_bench.datasets import REGISTRY
    from epde_bench.identity import problem_metadata
    rows = []
    for name, spec in REGISTRY.items():
        meta = problem_metadata(name)
        rows.append({'name': name, 'title': spec.title, 'suite': spec.suite,
                     'class': KIND_NAMES.get(spec.kind, spec.kind), 'kind': spec.kind,
                     'source': spec.source, 'law known': meta['truth_known'], 'files': spec.files})
    return rows


def dataset_names(with_truth=False, kinds=None, suites=None):
    rows = registry_table()
    return [r['name'] for r in rows
            if (not with_truth or r['law known'])
            and (kinds is None or r['kind'] in kinds)
            and (suites is None or r['suite'] in suites)]


def dataset_label(name):
    titles = {r['name']: r['title'] for r in registry_table()}
    return titles.get(name, name)


VARIANT_LABELS = {
    'default': 'EPDE, shipped settings',
    'legacy': 'EPDE, earlier settings (squared error, LASSO, complexity)',
    'poly': 'EPDE, polynomial derivatives',
    'knee': 'EPDE, sparsity at the knee of the error curve',
    'instab_cv': 'EPDE, instability by cross-validation',
    'trig_native': "EPDE, library's sine and cosine on a narrow frequency interval",
    'pysindy': 'sparse regression (PySINDy)',
    'custom': 'EPDE, settings from the upload page',
}


def variant_label(name):
    return VARIANT_LABELS.get(name, name)


@st.cache_resource(show_spinner='Loading data…')
def load_problem(name, loader_kwargs_json='{}'):
    from epde_bench.datasets import load
    return load(name, **json.loads(loader_kwargs_json))


def variants():
    from epde_bench.config import variants as _variants
    return list(_variants())


def search_config(name, variant='default', overrides=None):
    from epde_bench.config import load_config, resolve_for_problem
    cfg = load_config(name, variant, overrides)
    problem = load_problem(name, json.dumps(cfg.get('loader', {}), sort_keys=True))
    return cfg, problem, resolve_for_problem(cfg, problem)


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def show_figure(fig):
    import matplotlib.pyplot as plt
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def cpu_count():
    return os.cpu_count() or 1


def link(path, label, icon=None):
    """``st.page_link`` that degrades to plain text when the page is run on its own."""
    try:
        st.page_link(path, label=label, icon=icon)
    except Exception:          # noqa: BLE001  (StreamlitPageNotFoundError outside the multipage app)
        st.markdown(f'{icon or ""} {label}')
