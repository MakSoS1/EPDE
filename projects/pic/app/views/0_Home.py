"""Quiet, actionable entrance to the existing six-page EPDE Explorer."""
from __future__ import annotations

import streamlit as st

from support.discovery_lab import job_queue as queue
from support.discovery_lab.session import current_owner
from support.discovery_lab.catalog import full_catalog
from support.minimal_ui import intro

intro("EPDE Explorer", "Discover governing equations from data with EPDE and PySINDy.")

a,b=st.columns([1,1.15],gap="small")
with a:
    if st.button("Start a search  →",type="primary",use_container_width=True):
        st.switch_page("views/10_Discovery_lab.py")
with b:
    if st.button("Browse data catalog  →",use_container_width=True):
        st.switch_page("views/14_Data_catalog.py")

st.write("")
st.subheader("How it works")
cols=st.columns(4,gap="small")
for col,num,title,caption in zip(
    cols,range(1,5),
    ("Choose data","Configure methods","Run or export","Review & compare"),
    ("Built-in or uploaded","Independent EPDE / PySINDy settings",
     "Light on server, heavy on your PC","Recovered equations and metrics")):
    with col:
        st.markdown(f"**{num}. {title}**")
        st.caption(caption)

st.write("")
owner=current_owner()
runs=queue.list_jobs(owner)
completed=sum(r.get("state")=="finished" for r in runs)
a,b,c=st.columns(3)
a.metric("Your experiments",len(runs))
b.metric("Completed",completed)
c.metric("Available datasets",len(full_catalog()))

st.subheader("Continue")
actions=st.columns(3,gap="medium")
with actions[0]:
    with st.container(border=True):
        st.markdown("**New experiment**")
        st.caption("Choose data and run a scientific discovery method.")
        if st.button("Run a search",key="home_run",use_container_width=True):
            st.switch_page("views/10_Discovery_lab.py")
with actions[1]:
    with st.container(border=True):
        st.markdown("**Browse equations**")
        st.caption("Find an ODE, PDE or dataset from the complete catalog.")
        if st.button("Open catalog",key="home_catalog",use_container_width=True):
            st.switch_page("views/14_Data_catalog.py")
with actions[2]:
    with st.container(border=True):
        st.markdown("**Import local results**")
        st.caption("View heavy PDE results and compare them with prior runs.")
        if st.button("Open Results",key="home_results",use_container_width=True):
            st.switch_page("views/11_Discovery_results.py")

if runs:
    st.subheader("Recent experiments")
    st.dataframe([{
        "Experiment": r.get("label") or r.get("dataset"),
        "Method": {"epde":"EPDE","pdecomp":"PySINDy"}.get(r.get("method"),r.get("method")),
        "Status":r.get("state","unknown"),
    } for r in runs[:5]],hide_index=True,use_container_width=True)
    if st.button("See all results",key="home_see_all"):
        st.switch_page("views/11_Discovery_results.py")

with st.container(border=True):
    st.markdown("**Large PDEs run locally**")
    st.write("Choose a heavy equation in Run a search, adjust the scientific settings, "
             "download experiment.zip and the repository, and use the included launcher. "
             "Upload experiment-results.zip in Results to compare matching inputs. "
             "Large data are never loaded by the hosted website.")
