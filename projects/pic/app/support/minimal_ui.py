"""Shared theme-aware minimalist Streamlit UI."""
import streamlit as st

BASE = """
<style>
html, body, [class*="css"] {font-family:Inter,system-ui,-apple-system,"Segoe UI",sans-serif}
.block-container {padding-top:1.35rem;padding-bottom:2.8rem;max-width:1350px}
[data-testid="stSidebar"] .stMarkdown p {font-size:.88rem}
h1 {font-size:2rem!important;font-weight:720!important;letter-spacing:-.035em!important;margin:.2rem 0 .3rem!important}
h2 {font-size:1.27rem!important;font-weight:680!important;letter-spacing:-.02em!important}
h3 {font-size:1.05rem!important;font-weight:650!important}
[data-testid="stVerticalBlockBorderWrapper"] > div[data-testid="stVerticalBlock"] {border-radius:14px}
[data-testid="stMetric"] {padding:11px 16px;border:1px solid var(--epde-border);background:var(--epde-surface);border-radius:12px}
[data-testid="stMetricLabel"] {color:var(--epde-muted)}
[data-testid="stMetricValue"] {font-size:1.45rem}
.stButton button,.stDownloadButton button,.stLinkButton a {border-radius:9px;min-height:2.45rem;font-weight:550}
[data-testid="stExpander"] {background:var(--epde-surface);border:1px solid var(--epde-border)!important;border-radius:11px}
[data-testid="stTabs"] [data-baseweb="tab-list"] {gap:6px;border-bottom:1px solid var(--epde-border)}
[data-testid="stTabs"] button[role="tab"] {font-weight:560}
[data-testid="stAlert"],[data-testid="stDataFrame"] {border-radius:10px}
.epde-subtitle {font-size:.97rem;color:var(--epde-muted);margin:-.1rem 0 1.05rem}
.epde-step {font-size:1.05rem;font-weight:660;color:var(--epde-heading);margin:.45rem 0 .8rem}
.epde-number {display:inline-flex;align-items:center;justify-content:center;background:var(--epde-step-bg);
color:var(--epde-step-text);width:28px;height:28px;border-radius:50%;margin-right:9px}
.epde-card {background:var(--epde-surface);border:1px solid var(--epde-border);padding:16px;border-radius:14px;margin-bottom:10px}
.epde-stagebar {display:flex;gap:8px;flex-wrap:wrap;margin:.6rem 0 1.2rem}
.epde-stageitem {display:flex;align-items:center;gap:8px;flex:1;min-width:135px;border-bottom:3px solid var(--epde-border);
padding:8px 0;font-size:.88rem;color:var(--epde-muted)}
.epde-stageitem.active {color:var(--epde-heading);border-bottom-color:var(--epde-accent);font-weight:660}
.epde-stageitem.done {border-bottom-color:var(--epde-accent)}
@media(max-width:900px) {.block-container {padding:1rem .8rem 2rem} h1 {font-size:1.7rem!important}
.epde-stageitem {min-width:90px;font-size:.79rem}}
</style>
"""
LIGHT = """
<style>
:root {--epde-border:#dfe8f4;--epde-muted:#657892;--epde-surface:#fff;
--epde-heading:#19365c;--epde-step-bg:#e4efff;--epde-step-text:#1769ed;--epde-accent:#1769ed}
.stApp {background:#fafcff;color:#152840}
[data-testid="stSidebar"] {background:#f4f8fd;border-right:1px solid #e5ecf6}
</style>
"""
DARK = """
<style>
:root {--epde-border:#30425a;--epde-muted:#a6b7cc;--epde-surface:#192538;
--epde-heading:#e8f2ff;--epde-step-bg:#233e67;--epde-step-text:#9fc4ff;--epde-accent:#7eb2ff}
.stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"] {background:#101a29!important;color:#e8f0fa!important}
[data-testid="stSidebar"],[data-testid="stSidebarContent"] {background:#152235!important;border-right:1px solid #30425a}
.stApp .stMarkdown,.stApp [data-testid="stMarkdownContainer"],.stApp p,.stApp label,
.stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp .stCaptionContainer,
.stApp [data-testid="stText"],.stApp [data-testid="stWidgetLabel"] {color:#e8f0fa!important}
.stApp [data-testid="stCaptionContainer"] p {color:#a6b7cc!important}
[data-testid="stVerticalBlockBorderWrapper"] {background:transparent}
[data-testid="stVerticalBlockBorderWrapper"] > div {border-color:#30425a!important}
[data-testid="stExpander"],[data-testid="stMetric"] {background:#192538!important;border-color:#30425a!important}
.stApp [data-baseweb="input"],.stApp [data-baseweb="select"],.stApp [data-baseweb="textarea"],
.stApp [data-baseweb="base-input"],.stApp [data-baseweb="popover"] {background:#1c2c43!important;color:#f2f6ff!important}
.stApp input,.stApp textarea,.stApp [role="combobox"] {color:#f2f6ff!important}
.stApp .stButton button[kind="secondary"],.stApp .stDownloadButton button[kind="secondary"],
.stApp .stLinkButton a {background:#20324b!important;color:#e9f3ff!important;border-color:#425777!important}
.stApp [data-testid="stDataFrame"] {color:#e9f3ff}
</style>
"""

def apply_ui():
    theme = st.session_state.get("epde_appearance", "Light")
    st.markdown(BASE + (DARK if theme == "Dark" else LIGHT),
                unsafe_allow_html=True)

def intro(title, subtitle):
    st.title(title)
    st.markdown('<div class="epde-subtitle">' + subtitle + '</div>',
                unsafe_allow_html=True)

def step(number, title):
    st.markdown(
        f'<div class="epde-step"><span class="epde-number">{number}</span>{title}</div>',
        unsafe_allow_html=True)

def wizard_progress(current):
    titles=("Methods & data", "Settings", "Review & run")
    items=''.join(
        f'<div class="epde-stageitem {"active" if i == current else "done" if i < current else ""}">'
        f'{i}. {title}</div>' for i,title in enumerate(titles,1))
    st.markdown('<div class="epde-stagebar">'+items+'</div>',unsafe_allow_html=True)
