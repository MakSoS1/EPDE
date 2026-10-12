"""EPDE explorer: a visual front end to EPDE and the projects/pic data sets.

    streamlit run projects/pic/app/Home.py

This file only builds the grouped navigation; the start page is ``overview.py``.
"""

import sys
from pathlib import Path

# Community Cloud starts from the repository root, rather than projects/pic.
APP_DIR = Path(__file__).resolve().parent
for path in (APP_DIR, APP_DIR.parent, APP_DIR.parents[2]):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from epde_bench import env

env.pin_blas_threads(1)

import streamlit as st

st.set_page_config(page_title='EPDE explorer', page_icon='🧮', layout='wide')

PAGES = {
    'Start': [
        st.Page('overview.py', title='Overview', icon='🧮', default=True),
        st.Page('views/9_How_EPDE_works.py', title='How EPDE works', icon='🧭'),
    ],
    'Explore data': [
        st.Page('views/1_Data_sets.py', title='Data sets', icon='🗂️'),
        st.Page('views/2_Signal_check.py', title='Signal check', icon='🔍'),
        st.Page('views/3_Derivatives.py', title='Derivatives', icon='📈'),
    ],
    'Find equations': [
        st.Page('views/5_Run_search.py', title='Run a search', icon='🚀'),
        st.Page('views/6_Your_data.py', title='Your data', icon='📤'),
    ],
    'Review': [
        st.Page('views/7_Results.py', title='Results', icon='🗃️'),
        st.Page('views/8_Benchmark.py', title='Benchmark', icon='🏁'),
    ],
}

st.navigation(PAGES).run()
