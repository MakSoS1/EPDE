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
/*
 * Streamlit runs with a light native theme on Cloud. Changing the page
 * background alone leaves light BaseWeb buttons and selection widgets.
 * The following complete control palette applies ONLY when Dark is selected
 * in the EPDE Explorer sidebar; Light retains the genuine Streamlit theme.
 */
:root {
  --epde-border:#364a64;--epde-muted:#aec0d6;--epde-surface:#1b293d;
  --epde-heading:#edf5ff;--epde-step-bg:#29466b;--epde-step-text:#b5d5ff;
  --epde-accent:#83b8ff;
  --epde-dark-bg:#101a29;--epde-dark-sidebar:#152235;
  --epde-dark-control:#213249;--epde-dark-hover:#2c4362;
  --epde-dark-selected:#254d80;--epde-dark-stroke:#49607b;
  --epde-dark-text:#edf5ff;--epde-dark-subtext:#b5c5db;
}
html, body, .stApp,
[data-testid="stAppViewContainer"],[data-testid="stMain"],
[data-testid="stMainBlockContainer"] {
  background-color:var(--epde-dark-bg)!important;
  color:var(--epde-dark-text)!important;
}
[data-testid="stSidebar"],[data-testid="stSidebarContent"],
[data-testid="stSidebarUserContent"] {
  background-color:var(--epde-dark-sidebar)!important;
  color:var(--epde-dark-text)!important;
  border-color:var(--epde-border)!important;
}
/* The bright top toolbar in a light Streamlit deployment was a major
   contrast break on the user's screenshot. */
[data-testid="stHeader"],header[data-testid="stHeader"],
[data-testid="stToolbar"],[data-testid="stToolbarActions"],
[data-testid="stAppDeployButton"] {
  background-color:var(--epde-dark-bg)!important;
  color:var(--epde-dark-text)!important;
}
[data-testid="stHeader"] button,[data-testid="stToolbar"] button {
  color:var(--epde-dark-text)!important;
}
[data-testid="stHeader"] svg,[data-testid="stToolbar"] svg {
  fill:currentColor;
  color:var(--epde-dark-text)!important;
}
/* General content and cards */
.stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stMarkdownContainer"] li,
.stApp [data-testid="stText"],.stApp label,
.stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp h5 {
  color:var(--epde-dark-text)!important;
}
.stApp [data-testid="stCaptionContainer"],
.stApp [data-testid="stCaptionContainer"] p,
.stApp .epde-subtitle,.stApp .epde-stageitem,
.stApp .epde-muted {color:var(--epde-dark-subtext)!important}
.stApp .epde-stageitem.active {color:var(--epde-dark-text)!important}
.stApp [data-testid="stVerticalBlockBorderWrapper"] > div,
.stApp [data-testid="stExpander"],
.stApp [data-testid="stMetric"] {
  border-color:var(--epde-border)!important;
}
.stApp [data-testid="stExpander"],
.stApp [data-testid="stMetric"] {
  background:var(--epde-surface)!important;
}
.stApp [data-testid="stAlert"] {background:#203049!important}
.stApp [data-testid="stTabs"] [role="tab"] {
  color:var(--epde-dark-subtext)!important;
}
.stApp [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
  color:var(--epde-dark-text)!important;
  border-bottom-color:var(--epde-accent)!important;
}
/* All buttons used by st.pills and st.segmented_control. Streamlit's actual
   BaseButton kind/testid matters more than the outer widget class. */
.stApp button[kind="pills"],
.stApp button[kind="segmented_control"],
.stApp [data-testid="stBaseButton-pills"],
.stApp [data-testid="stBaseButton-segmented_control"] {
  background:var(--epde-dark-control)!important;
  color:var(--epde-dark-text)!important;
  border:1px solid var(--epde-dark-stroke)!important;
  box-shadow:none!important;
}
.stApp button[kind="pillsActive"],
.stApp button[kind="segmented_controlActive"],
.stApp [data-testid="stBaseButton-pillsActive"],
.stApp [data-testid="stBaseButton-segmented_controlActive"],
.stApp [data-testid="stButtonGroup"] button[aria-pressed="true"] {
  background:var(--epde-dark-selected)!important;
  color:#ffffff!important;
  border:1px solid #7db4ff!important;
  box-shadow:inset 0 0 0 1px #4e86d4!important;
}
.stApp button[kind^="pills"]:not(:disabled):hover,
.stApp button[kind^="segmented_control"]:not(:disabled):hover {
  filter:brightness(1.17);
}
.stApp button[kind^="pills"] *,
.stApp button[kind^="segmented_control"] * {
  color:inherit!important;
}
/* Streamlit >=1.5x renders pills/segmented controls as radio buttons
   inside stButtonGroup, without BaseButton kind or data-testid on each option.
   The old selectors above intentionally remain as compatibility fallbacks. */
.stApp [data-testid="stButtonGroup"] button[role="radio"],
.stApp [data-testid="stButtonGroup"] button[aria-pressed="false"] {
  background:var(--epde-dark-control)!important;
  color:var(--epde-dark-text)!important;
  border:1px solid var(--epde-dark-stroke)!important;
  box-shadow:none!important;
}
.stApp [data-testid="stButtonGroup"] button[role="radio"][aria-checked="true"],
.stApp [data-testid="stButtonGroup"] button[aria-pressed="true"] {
  background:var(--epde-dark-selected)!important;
  color:#ffffff!important;
  border-color:#7db4ff!important;
  box-shadow:inset 0 0 0 1px #4e86d4!important;
}
.stApp [data-testid="stButtonGroup"] button[role="radio"]:not(:disabled):hover,
.stApp [data-testid="stButtonGroup"] button[aria-pressed]:not(:disabled):hover {
  filter:brightness(1.17);
}
.stApp [data-testid="stButtonGroup"] button[role="radio"] *,
.stApp [data-testid="stButtonGroup"] button[aria-pressed] * {
  color:inherit!important;
}
/* Standard action buttons, links, nav and sidebar selection. */
.stApp [data-testid="stBaseButton-secondary"],
.stApp .stButton button[kind="secondary"],
.stApp .stDownloadButton button[kind="secondary"],
.stApp .stLinkButton a {
  background:var(--epde-dark-control)!important;
  color:var(--epde-dark-text)!important;
  border:1px solid var(--epde-dark-stroke)!important;
}
.stApp [data-testid="stBaseButton-primary"],
.stApp .stButton button[kind="primary"],
.stApp .stDownloadButton button[kind="primary"] {
  background:#397ce0!important;color:#ffffff!important;
  border-color:#397ce0!important;
}
.stApp button:disabled {
  opacity:.60!important;
  cursor:not-allowed!important;
}
.stApp [data-testid="stSidebarNav"] a {
  color:var(--epde-dark-text)!important;
}
.stApp [data-testid="stSidebarNav"] a[aria-current="page"],
.stApp [data-testid="stSidebarNav"] a[aria-selected="true"] {
  background:#2d486d!important;color:#ffffff!important;
}
/* Input controls (BaseWeb): div is the painted surface, not the input.
   Covers selectbox, multiselect, numeric/text inputs and text areas. */
.stApp [data-testid="stSelectbox"] [data-baseweb="select"] > div,
.stApp [data-testid="stMultiSelect"] [data-baseweb="select"] > div,
.stApp [data-baseweb="select"] > div,
.stApp [data-baseweb="base-input"],
.stApp [data-baseweb="input"] > div,
.stApp [data-baseweb="textarea"] > div,
.stApp [data-testid="stNumberInput"] > div,
.stApp [data-testid="stTextInput"] > div {
  background:var(--epde-dark-control)!important;
  border-color:var(--epde-dark-stroke)!important;
  color:var(--epde-dark-text)!important;
}
.stApp input,.stApp textarea,
.stApp [role="combobox"],.stApp [data-baseweb="select"] span,
.stApp [data-baseweb="select"] svg,
.stApp [data-baseweb="input"] svg,
.stApp [data-testid="stNumberInput"] button {
  color:var(--epde-dark-text)!important;
  caret-color:#ffffff!important;
}
.stApp input::placeholder,.stApp textarea::placeholder {
  color:#afbed2!important;opacity:1!important;
}
/* Newer Selectbox is an input[role=combobox] rather than BaseWeb select.
   Color the input AND each painted direct wrapper to remove the white field. */
.stApp [data-testid="stSelectbox"] input[role="combobox"],
.stApp [data-testid="stSelectbox"] div:has(> input[role="combobox"]),
.stApp [data-testid="stSelectbox"] div:has(> div > input[role="combobox"]),
.stApp [data-testid="stSelectbox"] div:has(> div > div > input[role="combobox"]) {
  background-color:var(--epde-dark-control)!important;
  color:var(--epde-dark-text)!important;
  border-color:var(--epde-dark-stroke)!important;
}
.stApp [data-testid="stSelectbox"] button,
.stApp [data-testid="stSelectbox"] button svg {
  color:var(--epde-dark-text)!important;
}
.stApp [data-testid="stNumberInput"] button,
.stApp [data-testid="stFileUploader"] section,
.stApp [data-testid="stFileUploader"] button {
  background:var(--epde-dark-control)!important;
  color:var(--epde-dark-text)!important;
  border-color:var(--epde-dark-stroke)!important;
}
.stApp [data-baseweb="tag"] {
  background:#315276!important;color:#ffffff!important;
}
/* Select popovers are rendered into a body portal, outside .stApp. */
body [data-baseweb="popover"],
body [data-baseweb="menu"],body [role="listbox"],
body [role="listbox"] ul {
  background:var(--epde-dark-control)!important;
  color:var(--epde-dark-text)!important;
  border-color:var(--epde-dark-stroke)!important;
}
body [role="option"],body [role="listbox"] li,
body [data-baseweb="menu"] li {
  background:var(--epde-dark-control)!important;
  color:var(--epde-dark-text)!important;
}
body [role="option"]:hover,body [role="option"][aria-selected="true"],
body [role="listbox"] li:hover,
body [data-baseweb="menu"] li:hover {
  background:var(--epde-dark-selected)!important;
  color:#ffffff!important;
}
/* Code, markdown tables and DataFrame containers */
.stApp [data-testid="stCodeBlock"] pre,
.stApp pre,
.stApp code:not([class*="language-"]) {
  background-color:#1b2b40!important;
  border-color:var(--epde-dark-stroke)!important;
}
.stApp [data-testid="stDataFrame"],
.stApp [data-testid="stTable"] {
  background:var(--epde-surface)!important;
  color:var(--epde-dark-text)!important;
}
.stApp [data-testid="stToggle"] label,
.stApp [data-testid="stCheckbox"] label {
  color:var(--epde-dark-text)!important;
}
/* Visible keyboard focus in both themes; do not sacrifice accessibility. */
.stApp :is(button,input,textarea,[role="combobox"]):focus-visible {
  outline:2px solid #92c2ff!important;
  outline-offset:2px;
}
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
