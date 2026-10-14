"""EPDE explorer. Research pages remain available in local mode.

Set EPDE_PUBLIC_DEMO=1 for the isolated visitor-facing Discovery Lab.
This setting is intentionally opt-in; legacy pages have shared job files.
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
st.set_page_config(page_title="EPDE Explorer", page_icon="🧮", layout="wide")

if os.environ.get("EPDE_PUBLIC_DEMO") == "1":
    # Do not expose old shared-instance Results, Jobs, Benchmark, Upload or Overview.
    st.navigation([st.Page("views/10_Discovery_lab.py", title="Equation Discovery Lab",
                           icon="🧪", default=True)]).run()
else:
    PAGES = {
        "Experiments": [
            st.Page("views/10_Discovery_lab.py", title="Equation Discovery Lab", icon="🧪", default=True),
        ],
        "Start": [
            st.Page("overview.py", title="Overview", icon="🧮"),
            st.Page("views/9_How_EPDE_works.py", title="How EPDE works", icon="🧭"),
        ],
        "Explore data": [
            st.Page("views/1_Data_sets.py", title="Data sets", icon="🗂️"),
            st.Page("views/2_Signal_check.py", title="Signal check", icon="🔍"),
            st.Page("views/3_Derivatives.py", title="Derivatives", icon="📈"),
        ],
        "Find equations": [
            st.Page("views/5_Run_search.py", title="Run a search (legacy)", icon="🚀"),
            st.Page("views/4_SINDy.py", title="SINDy baseline", icon="🧩"),
            st.Page("views/6_Your_data.py", title="Your data (legacy)", icon="📤"),
        ],
        "Review": [
            st.Page("views/7_Results.py", title="Results (legacy)", icon="🗃️"),
            st.Page("views/8_Benchmark.py", title="Benchmark (legacy)", icon="🏁"),
        ],
    }
    st.navigation(PAGES).run()
