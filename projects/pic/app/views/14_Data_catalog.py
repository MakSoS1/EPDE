"""Minimal, searchable presentation of the complete metadata-only PDE registry.

No scientific arrays are loaded by browsing. The original source and all 59+
records remain available; availability and compatibility are never invented.
"""
from __future__ import annotations

import inspect
import math

import streamlit as st

from support.minimal_ui import intro
from support.discovery_lab.catalog import (
    declared_equations, full_catalog, source_file_status,
)
from support.discovery_lab.execution_policy import is_local_only

intro("Data catalog", "Search the complete scientific registry and choose an equation.")
@st.cache_data(show_spinner=False)
def metadata():
    return full_catalog()

records = metadata()
search = st.text_input("Search dataset or equation",
                       placeholder="Navier–Stokes, Burgers, Lorenz, ...")
filters = st.pills("Dataset type", ["All", "ODE", "PDE", "Real data", "Heavy"],
                   default="All", selection_mode="single") or "All"

def included(row):
    kind = row["kind"]
    if filters == "ODE" and not kind.startswith("ode"):
        return False
    if filters == "PDE" and not kind.startswith("pde"):
        return False
    if filters == "Real data" and row["suite"] != "real":
        return False
    if filters == "Heavy" and not is_local_only(row["id"]):
        return False
    words = " ".join(str(row[x]) for x in ("id", "title", "source", "files"))
    return not search or search.casefold() in words.casefold()

filtered = [row for row in records if included(row)]
st.caption(f"Showing {len(filtered)} of {len(records)} registered scientific datasets.")
if not filtered:
    st.info("No matching equations. Try another search or dataset type.")
    st.stop()

selected = st.session_state.get("catalog_selected")
available = {r["id"] for r in filtered}
if selected not in available:
    selected = filtered[0]["id"]
    st.session_state["catalog_selected"] = selected

left, right = st.columns([2.35, 1.15], gap="large")
with left:
    page_size = 6
    total_pages = math.ceil(len(filtered) / page_size)
    idx = min(st.session_state.get("catalog_page", 0), total_pages-1)
    st.session_state["catalog_page"] = idx
    view = filtered[idx*page_size:(idx+1)*page_size]
    for offset in range(0,len(view),2):
        cols = st.columns(2,gap="small")
        for col,row in zip(cols,view[offset:offset+2]):
            with col:
                with st.container(border=True):
                    st.markdown("**"+row["title"]+"**")
                    kind = "ODE" if row["kind"].startswith("ode") else "PDE"
                    st.caption(kind+" · "+row["suite_label"]+" · "+
                               ("Local computation" if is_local_only(row["id"])
                                else "Hosted if within limits"))
                    if st.button("View details  →", key="catalog_card_"+row["id"],
                                 type="primary" if row["id"]==selected else "secondary",
                                 use_container_width=True):
                        st.session_state["catalog_selected"] = row["id"]
                        st.rerun()
    prev, current, nex = st.columns([1,2,1])
    with prev:
        if st.button("← Previous", disabled=idx==0,
                     use_container_width=True, key="catalog_prev"):
            st.session_state["catalog_page"]=idx-1
            st.rerun()
    with current:
        st.caption(f"Page {idx+1} of {total_pages}")
    with nex:
        if st.button("Next →", disabled=idx==total_pages-1,
                     use_container_width=True, key="catalog_next"):
            st.session_state["catalog_page"]=idx+1
            st.rerun()

row = next(r for r in filtered if r["id"]==selected)
with right:
    with st.container(border=True):
        st.markdown("**Dataset details**")
        st.subheader(row["title"])
        st.caption(row["kind_label"]+" · "+row["suite_label"])
        st.write("**Source:**", row["source"])
        st.write("**Scientific ID:**", "`"+selected+"`")
        st.write("**Execution:**", "Local only · heavy" if is_local_only(selected)
                 else "Hosted within safety limits")
        from epde_bench.baselines import unsupported_reason
        reason = unsupported_reason(selected)
        st.write("**Available methods:**", "EPDE" + (", PySINDy" if not reason else ""))
        if reason:
            st.caption("PySINDy adapter unsupported: " + reason)
        state = source_file_status(selected)
        if state["status"]=="missing":
            st.warning("Some original scientific data files are missing. "
                       "See DATASETS.md before attempting a local run.")
        elif state["status"]=="unknown":
            st.caption("Original data availability cannot be verified statically.")
        if st.button("Use this record in Lab", type="primary", use_container_width=True):
            st.session_state["catalog_requested_record"]=selected
            st.switch_page("views/10_Discovery_lab.py")
    with st.expander("Original equation and loader details", expanded=False):
        laws = declared_equations(selected)
        if laws:
            st.markdown("**Declared governing equation(s)**")
            for law in laws:
                st.code(law, language=None)
        else:
            st.caption("This record has no statically declared reference law.")
        st.caption("Original registered source files: " + str(row["files"]))
        from epde_bench.datasets import REGISTRY
        st.code(inspect.getsource(REGISTRY[selected].loader), language="python")

st.caption("Catalog entries describe scientific source records; not all external data "
           "files are included with the repository. Browsing never loads heavy arrays.")
