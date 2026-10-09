"""Interactive architecture diagrams from the LikeC4 model in ``projects/pic/architecture``.

The views are bundled into one self-contained script (regenerate it after editing the model):

    cd projects/pic/architecture
    likec4 gen webcomponent -o ../app/static/likec4-views.js .
"""

import html

import streamlit as st
import streamlit.components.v1 as components

from .common import APP_DIR

BUNDLE = APP_DIR / 'static' / 'likec4-views.js'


@st.cache_resource(show_spinner=False, max_entries=1)
def _read_bundle(modified):
    # keyed by the modification time, so a regenerated script is picked up without a restart
    return BUNDLE.read_text(encoding='utf-8').replace('</script', '<\\/script')


def _bundle():
    return _read_bundle(BUNDLE.stat().st_mtime_ns)


def available():
    return BUNDLE.exists()


def likec4_view(view_id, height=480, sequence=False):
    """One view; clicking it opens the full model browser to zoom and drill down."""
    variant = ' dynamic-variant="sequence"' if sequence else ''
    page = (f'<script>{_bundle()}</script>'
            f'<likec4-view view-id="{html.escape(view_id)}" browser="true"{variant} '
            f'style="display:block;width:100%;height:{height - 10}px"></likec4-view>')
    components.html(page, height=height)
