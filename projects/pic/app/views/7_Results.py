from pathlib import Path

import pandas as pd
import streamlit as st

from support.common import RESULTS_DIR, dataset_label, page, read_json, variant_label
from support.viewer import render_record

page('Results', '🗃️')
st.caption('Every saved search: from this app, the notebooks, the campaigns and the command line. '
           'Click a row to open it.')


@st.cache_data(ttl=30, show_spinner='Reading records…')
def index(root):
    rows = []
    for path in Path(root).rglob('*.json'):
        if path.name in ('campaign.json',) or 'jobs' in path.parts or path.name.endswith('.pending.json'):
            continue
        rec = read_json(path)
        if not isinstance(rec, dict) or 'status' not in rec or 'dataset' not in rec:
            continue
        m = rec.get('metrics') or {}
        parts = path.relative_to(root).parts
        rows.append({'source': parts[0] if len(parts) > 1 else '.',
                     'dataset': dataset_label(rec['dataset']) if rec['dataset'] != 'custom' else 'uploaded data',
                     'variant': variant_label(rec.get('variant')), 'noise': rec.get('noise'), 'seed': rec.get('seed'),
                     'status': rec['status'], 'truth on front': m.get('success_front'),
                     'pick correct': m.get('success_selected'), 'fit, s': rec.get('fit_seconds'),
                     'started': rec.get('started'), 'path': str(path)})
    return pd.DataFrame(rows)


df = index(str(RESULTS_DIR))
if df.empty:
    st.info(f'No records under {RESULTS_DIR} yet.')
    st.stop()
SOURCES = {'app': 'this app', 'notebook_cache': 'notebooks'}
df['from'] = df.source.map(lambda s: SOURCES.get(s, 'campaign or command line'))
c1, c2 = st.columns([3, 1], vertical_alignment='bottom')
datasets = c1.multiselect('record', sorted(df.dataset.unique()), placeholder='all records')
failed = c2.toggle('failed runs', value=False)
view = df if failed else df[df.status.isin(['ok'])]
if datasets:
    view = view[view.dataset.isin(datasets)]
view = view.sort_values('started', ascending=False)
columns = ['dataset', 'variant', 'noise', 'seed', 'truth on front', 'pick correct', 'fit, s', 'from']
if failed:
    columns.insert(4, 'status')
picked = st.dataframe(view[columns].rename(columns={'dataset': 'record', 'variant': 'method'}),
                      hide_index=True, width='stretch', on_select='rerun', selection_mode='single-row')
rows = picked.selection.rows if picked and picked.selection else []
if rows:
    record = read_json(view.iloc[rows[0]]['path'])
    st.divider()
    render_record(record, key='results')
