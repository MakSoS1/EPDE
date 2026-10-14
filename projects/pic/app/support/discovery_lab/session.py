"""Shared, capability-scoped session across Lab, Results and Compare.

A session ID is a secret recovery capability, not login/authentication.
The same ID is retained in Streamlit session_state during page navigation
and restored from the URL when someone reloads a bookmarked page.
"""
from __future__ import annotations

import streamlit as st

from . import job_queue as queue


def current_owner() -> str:
    from_url = st.query_params.get("session", "")
    from_state = st.session_state.get("discovery_lab_owner", "")
    owner = None
    for candidate in (from_url, from_state):
        try:
            queue.owner_path(candidate)
            owner = candidate
            break
        except (ValueError, TypeError):
            pass
    if owner is None:
        owner = queue.new_owner()
    st.session_state["discovery_lab_owner"] = owner
    # Streamlit clears query parameters on page navigation; restore for
    # bookmarks and hard reloads. Avoid advertising the token in result exports.
    if st.query_params.get("session") != owner:
        st.query_params["session"] = owner

    with st.sidebar.expander("Your saved session"):
        st.caption("Results stay associated with this private session code. "
                   "Bookmark the page or save the code to return after a reload. "
                   "Anyone with this link can access these results.")
        st.code(owner, language=None)
        saved = st.text_input("Open saved session code", type="password",
                              key="discovery_restore_code")
        if st.button("Restore", key="discovery_restore"):
            try:
                recovered = saved.strip()
                if not queue.owner_path(recovered).is_dir():
                    raise ValueError("No saved session")
                st.session_state["discovery_lab_owner"] = recovered
                st.query_params["session"] = recovered
                st.rerun()
            except (TypeError, ValueError):
                st.error("Session code not found.")
    return owner
