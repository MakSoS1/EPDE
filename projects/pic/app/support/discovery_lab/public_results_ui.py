"""Explicit navigation between private runs and immutable public examples."""
from __future__ import annotations
import streamlit as st
from . import public_examples


_COLLECTION_KEY = "discovery_result_collection"
_WIDGET_KEY = "_discovery_result_collection_widget"
_COLLECTIONS = ["My results", "Public results"]


def _remember_collection():
    selected = st.session_state.get(_WIDGET_KEY)
    if selected in _COLLECTIONS:
        st.session_state[_COLLECTION_KEY] = selected


def choose_collection(has_private):
    # Streamlit removes widgets when changing pages. Keep user intent in
    # a non-widget key, and rehydrate the page-local control on every render.
    preferred = st.session_state.get(_COLLECTION_KEY)
    if preferred not in _COLLECTIONS:
        preferred = "My results" if has_private else "Public results"
    st.session_state[_WIDGET_KEY] = preferred
    chosen = st.segmented_control(
        "Result collection", _COLLECTIONS, key=_WIDGET_KEY,
        selection_mode="single", on_change=_remember_collection,
    )
    st.session_state[_COLLECTION_KEY] = chosen if chosen in _COLLECTIONS else preferred
    return st.session_state[_COLLECTION_KEY] == "Public results"


def public_groups():
    st.caption('Read-only curated examples from real native experiments on builtin synthetic data. '
               'These examples teach the metrics; they are not a representative leaderboard. '
               'Your private runs are never published automatically.')
    try:return public_examples.load_library()
    except (ValueError,OSError,KeyError,TypeError):
        st.error('The public example library failed its integrity check. Private results remain available.')
        return []


def choose_example(groups):
    ids=[g['id'] for g in groups];mapping={g['id']:g for g in groups}
    preferred=st.session_state.get('public_example_group')
    chosen=st.selectbox('Public example',ids,index=ids.index(preferred) if preferred in ids else 0,
                       format_func=lambda k:mapping[k]['title'],key='public_example_selector')
    st.session_state['public_example_group']=chosen
    group=mapping[chosen]
    st.info(group['summary'])
    st.caption(group.get('selection_note','Fixed, educational selection; original equations and scores are retained.'))
    return group


def show_provenance(record):
    source=record.get('provenance') or {}
    st.caption('Public example · read-only · policy: '+
               str((record.get('advisor_experiment') or {}).get('label','recorded settings')))
    with st.expander('Public result provenance'):
        st.json(source)
        st.caption('Original search outputs are retained. Plot reconstruction, when present, '
                   'is an explicitly recorded post-processing step, not another search.')
        if record.get('evaluation_provenance'):
            st.json(record['evaluation_provenance'])
