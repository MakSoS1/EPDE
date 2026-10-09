"""Start page: what EPDE does and where to begin."""

import streamlit as st

from support import jobs
from support.common import link, page

page('EPDE explorer', '🧮')

st.markdown('EPDE finds the differential equation behind your data and shows how well it describes them. '
            'Pick a step below; every page explains itself in **About this page**.')

GROUPS = [
    ('1 · Explore data', [
        ('views/1_Data_sets.py', '🗂️', 'Data sets', 'records with known laws and measurements'),
        ('views/2_Signal_check.py', '🔍', 'Signal check', 'is the law still visible under noise?'),
        ('views/3_Derivatives.py', '📈', 'Derivatives', 'how noise affects derivatives'),
    ]),
    ('2 · Find equations', [
        ('views/5_Run_search.py', '🚀', 'Run a search', 'choose a record, press Start'),
        ('views/6_Your_data.py', '📤', 'Your data', 'upload a file and search it'),
    ]),
    ('3 · Review', [
        ('views/7_Results.py', '🗃️', 'Results', 'every saved search'),
        ('views/8_Benchmark.py', '🏁', 'Benchmark', 'compare methods on many records'),
    ]),
]
for column, (title, items) in zip(st.columns(3), GROUPS):
    with column:
        st.subheader(title)
        for path, icon, label, text in items:
            link(path, label, icon)
            st.caption(text)

st.divider()
link('views/9_How_EPDE_works.py', 'New to EPDE? See how it works', '🧭')

running = [j for j in jobs.jobs() if jobs.status(j) == 'running']
if running:
    st.info('Running now: ' + '; '.join(job['title'] for job in running))
