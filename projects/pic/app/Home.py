"""Equation Discovery Lab: one public Streamlit application, three clear pages.

The old teaching/research pages stay in the repository for regression and
private research, but are not offered to anonymous public visitors.
"""
import os
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
for path in (APP_DIR, APP_DIR.parent, APP_DIR.parents[2]):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from epde_bench import env
env.pin_blas_threads(1)

import streamlit as st

st.set_page_config(page_title="Equation Discovery Lab", page_icon="🧪", layout="wide")

PAGES = {
    "Workspace": [
        st.Page("views/10_Discovery_lab.py", title="Lab", icon="🧪", default=True),
        st.Page("views/11_Discovery_results.py", title="Results", icon="🗂️"),
        st.Page("views/12_Discovery_compare.py", title="Compare", icon="⚖️"),
    ],
    "Learn": [
        st.Page("views/9_How_EPDE_works.py", title="How it works · EPDE & SINDy", icon="🧭"),
    ],
}

# Preserve the old exploratory interface without exposing shared-instance
# uploads, jobs and results on the hosted public demonstration.
if (os.environ.get("EPDE_SHOW_RESEARCH_PAGES") == "1"
        and os.environ.get("EPDE_PUBLIC_DEMO") != "1"):
    PAGES["Research tools (local only)"] = [
        st.Page("overview.py", title="Overview", icon="🧮"),
        st.Page("views/1_Data_sets.py", title="Data sets", icon="🗂️"),
        st.Page("views/2_Signal_check.py", title="Signal check", icon="🔍"),
        st.Page("views/3_Derivatives.py", title="Derivatives", icon="📈"),
        st.Page("views/5_Run_search.py", title="Run a search (legacy)", icon="🚀"),
        st.Page("views/4_SINDy.py", title="SINDy baseline (legacy)", icon="🧩"),
        st.Page("views/6_Your_data.py", title="Your data (legacy)", icon="📤"),
        st.Page("views/7_Results.py", title="Results (legacy)", icon="🗃️"),
        st.Page("views/8_Benchmark.py", title="Benchmark (legacy)", icon="🏁"),
    ]

st.sidebar.caption("Start in Lab to run methods. Use Results and Compare to "
                   "review them, or How it works to learn EPDE and SINDy.")
st.navigation(PAGES).run()
