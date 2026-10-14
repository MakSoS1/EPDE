"""Browser PySINDy runs on the visitor's device. No Streamlit CPU is used.

The separately pinned public static frontend already passed an external
Chromium/Pyodide/PySINDy smoke. Its 5 ODE tasks are *not* falsely presented
as the original PDEComp/native adaptation of 40+ ODE/PDE tasks.
"""
from __future__ import annotations

import streamlit as st

from support.common import page

page("PySINDy · your CPU", "⚡")
st.title("PySINDy — run in your browser")
st.caption("No installer or Python setup. Scientific calculations run inside your "
           "browser's WebAssembly Web Worker; Streamlit only displays this page.")

st.info(
    "**What works:** authentic PySINDy 2.1.0 with STLSQ on five ODE examples "
    "and custom time-series CSV. Finite differences and Savitzky–Golay are "
    "supported. Your browser keeps the generated results in its own IndexedDB. "
    "This is a limited browser configuration, **not the native PDEComp protocol** "
    "and not full EPDE."
)
st.warning(
    "Your browser must be able to download Pyodide and its scientific packages. "
    "The page itself is delivered directly by this Streamlit site, which "
    "avoids a dependency on GitHub HTML CDNs in Russia. However some Russian "
    "networks may block Pyodide/PyPI dependency downloads. "
    "This does not prove the method failed — use a different available "
    "network or try the separately installed native execution. "
    "No VPN bypass or disabled browser security is necessary."
)
CDN_REVISION = "dd4ecbafbc17f084cf09f4680a7a2b1d43416b15"
BROWSER_URL = (
    "https://raw.githack.com/MakSoS1/EPDE/"
    + CDN_REVISION
    + "/projects/pic/browser_demo/index.html"
)
st.link_button("Open browser lab in a separate tab", BROWSER_URL)
st.caption("The embedded browser lab has its own Results and Compare. "
           "For now browser-local records are not transferred to the "
           "Streamlit server's Results page. Download their JSON to retain them.")
from support.discovery_lab.browser_inline import build_browser_html

# Trusted repository code, inlined into a first-party Streamlit iframe.
# Heavy Python/WebAssembly calculations happen inside the visitor browser.
try:
    st.iframe(build_browser_html(embedded=True), height=1210,
              alt="PySINDy 2.1.0 scientific lab computing on this browser's CPU")
except (OSError, ValueError, RuntimeError) as exc:
    st.error("The bundled browser scientific engine is unavailable in this deployment. "
             "Use the standalone browser version from the link above.")

st.divider()
st.markdown(
    "### About the original native PDEComp benchmark\n"
    "Vlad's [PDEComp native protocol]"
    "(https://github.com/MakSoS1/PDEComp/tree/native) also supports "
    "higher-order ODEs, systems and several PDEs through custom native libraries "
    "and dataset-specific optimizers. The embedded browser implementation "
    "does **not** claim equivalence to those adapters. "
    "DeepMoD and DISCOVER have independent, version-incompatible container "
    "environments and are not silently substituted by PySINDy."
)
