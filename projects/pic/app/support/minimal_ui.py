"""Shared minimal visual language for all EPDE Explorer Streamlit pages.

Only presentation: no changes to data loading, scientific defaults, or security.
All components are native Streamlit for desktop and mobile compatibility.
"""
import streamlit as st


CSS = """
<style>
:root {--epde-blue:#1769ed;--epde-border:#e5ecf6;--epde-muted:#667995}
html, body, [class*="css"] {font-family:Inter,system-ui,-apple-system,"Segoe UI",sans-serif}
.stApp {background:#fafcff;color:#142641}
.block-container {padding-top:1.4rem;padding-bottom:3rem;max-width:1360px}
[data-testid="stSidebar"] {background:#f5f8fc;border-right:1px solid #e6edf6}
[data-testid="stSidebar"] .stMarkdown p {font-size:.88rem}
h1 {font-size:2rem!important;font-weight:720!important;letter-spacing:-.04em!important;margin:.2rem 0 .3rem!important}
h2 {font-size:1.28rem!important;font-weight:680!important;letter-spacing:-.025em!important}
h3 {font-size:1.05rem!important;font-weight:650!important}
[data-testid="stCaptionContainer"] {color:var(--epde-muted);line-height:1.45}
[data-testid="stVerticalBlockBorderWrapper"] > div[data-testid="stVerticalBlock"] {
  border-radius:14px
}
[data-testid="stMetric"] {padding:11px 16px;background:#fff;border:1px solid var(--epde-border);border-radius:12px}
[data-testid="stMetricLabel"] {color:var(--epde-muted)}
[data-testid="stMetricValue"] {font-size:1.45rem}
.stButton button,.stDownloadButton button,.stLinkButton a {border-radius:9px;min-height:2.45rem;font-weight:550}
[data-testid="stExpander"] {background:#fff;border:1px solid var(--epde-border)!important;border-radius:11px}
[data-testid="stExpander"] summary {font-weight:550}
[data-testid="stTabs"] [data-baseweb="tab-list"] {gap:6px;border-bottom:1px solid var(--epde-border)}
[data-testid="stTabs"] button[role="tab"] {font-weight:560}
[data-testid="stAlert"] {border-radius:10px}
[data-testid="stDataFrame"] {border-radius:10px;overflow:hidden}
[data-testid="stForm"], [data-testid="stFormSubmitButton"] {border-radius:12px}
.epde-subtitle {font-size:.97rem;color:#667995;margin:-.1rem 0 1.1rem}
.epde-kicker {font-size:.78rem;color:#65809d;font-weight:650;text-transform:uppercase;letter-spacing:.04em}
.epde-card {background:#fff;border:1px solid #e5ecf6;padding:16px;border-radius:14px;margin-bottom:10px}
.epde-card strong {font-weight:660}
.epde-muted {color:#667995;font-size:.9rem}
.epde-step {font-size:.96rem;font-weight:650;color:#19365c}
.epde-number {display:inline-flex;align-items:center;justify-content:center;background:#e5efff;
color:#196bdf;width:28px;height:28px;border-radius:50%;margin-right:8px}
@media(max-width:900px) {
  .block-container {padding:1rem .85rem 2rem}
  h1 {font-size:1.65rem!important}
  .epde-card {padding:12px}
}
</style>
"""


def apply_ui():
    st.markdown(CSS, unsafe_allow_html=True)


def intro(title, subtitle):
    st.title(title)
    st.markdown('<div class="epde-subtitle">'+subtitle+'</div>', unsafe_allow_html=True)


def step(number, title):
    st.markdown(
        f'<div class="epde-step"><span class="epde-number">{number}</span>{title}</div>',
        unsafe_allow_html=True,
    )
