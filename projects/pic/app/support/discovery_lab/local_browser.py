"""Browser-to-local-worker bridge using Streamlit Components v2.

Native requests originate in the visitor's browser and go to 127.0.0.1,
NEVER to the Streamlit Community Cloud machine's loopback interface.
"""
from __future__ import annotations

from pathlib import Path
import streamlit as st

ASSETS = Path(__file__).resolve().parent / "local_browser_assets"
HTML = """<div class="epde-local">
 <strong>Compute on this computer</strong>
 <p>The installed EPDE worker runs the real algorithm. The cloud only hosts this page
 and saves your result after you approve the local run.</p>
 <div class="epde-buttons"><button class="epde-start">Run on my CPU</button>
 <button class="epde-check" type="button">Check worker</button></div>
 <p class="epde-status" role="status" aria-live="polite"></p>
</div>"""
CSS = """
.epde-local {padding:16px;border:1px solid var(--st-border-color,#d5d9df);
 border-radius:10px;font:inherit;}
.epde-local p {margin:7px 0 10px;color:var(--st-text-color,#404040);}
.epde-buttons {display:flex;gap:12px;flex-wrap:wrap}
.epde-buttons button {font:inherit;cursor:pointer;padding:8px 18px;border-radius:8px;
 border:1px solid var(--st-primary-color,#3768aa);background:transparent;
 color:var(--st-text-color,#202020)}
.epde-buttons button.epde-start {background:var(--st-primary-color,#3768aa);color:white}
.epde-buttons button:disabled {opacity:.6;cursor:wait}
.epde-status {min-height:25px;font-size:.91em}
"""
JS = (ASSETS / "local_bridge.js").read_text(encoding="utf-8")


@st.cache_resource(show_spinner=False)
def _get_renderer():
    return st.components.v2.component(
        "epde_user_cpu_bridge", html=HTML, css=CSS, js=JS,
    )


def local_bridge(*, job: dict, token: str, key: str):
    renderer = _get_renderer()
    return renderer(
        key=key,
        data={"job": job, "token": token},
        on_result_change=lambda: None,
        on_failed_change=lambda: None,
    )
