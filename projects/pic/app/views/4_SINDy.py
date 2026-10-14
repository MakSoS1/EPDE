"""Dedicated PySINDy entry point using the *existing* reproducible benchmark runner."""
import streamlit as st

from support import jobs
from support.common import dataset_label, dataset_names, page, link

page("SINDy", "🧩")
st.caption("Compare a fixed-library sparse-regression baseline (PySINDy) with EPDE. "
           "The baseline uses the existing shared record format and benchmark metrics.")
st.info("Unlike EPDE, this implementation selects from a fixed library. "
        "Unsupported physical models are reported as unsupported, not silently scored wrong. "
        "The same dataset/seed/noise must be used for fair comparisons.")

names = dataset_names(with_truth=True)
name = st.selectbox("record", names, index=(names.index("ode") if "ode" in names else 0),
                    format_func=dataset_label)
col1, col2 = st.columns(2)
noise = col1.number_input("noise, %", min_value=0.0, max_value=10.0,
                          value=0.0, step=0.5)
seed = int(col2.number_input("seed", min_value=0, max_value=999, value=0))

a, b = st.columns(2)
if a.button("Run PySINDy", type="primary"):
    job = jobs.start_run(name, "pysindy", float(noise), seed)
    st.session_state["run_job"] = job["id"]
    st.switch_page("views/5_Run_search.py")
if b.button("Run EPDE on the same record"):
    job = jobs.start_run(name, "default", float(noise), seed)
    st.session_state["run_job"] = job["id"]
    st.switch_page("views/5_Run_search.py")

st.markdown("### Compare methods fairly")
st.write("The Benchmark page can run both methods across the same datasets, noise levels "
         "and random seeds and show exact selected structures, truths on fronts and runtime.")
link("views/8_Benchmark.py", "Open multi-method Benchmark", "🏁")
st.caption("LLM discovery and other PDEComp frameworks were discussed as a future extension. "
           "They are not integrated until the team selects adapters, a common protocol "
           "and safe API quotas; this page does not make paid or unmetered LLM calls.")
