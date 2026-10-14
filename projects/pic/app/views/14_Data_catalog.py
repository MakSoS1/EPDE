"""Full 59+ dataset catalogue; metadata-only by default, never allocate PDE grids."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from support.common import page
from support.discovery_lab.catalog import KIND_LABELS, SUITE_LABELS, declared_equations, full_catalog
from support.discovery_lab.execution_policy import is_local_only

page("All equations & datasets", "📚")
st.caption("The complete registry is read directly from epde_bench.datasets.REGISTRY. "
           "Core, extended, measured, and additional scientific records are all shown. "
           "Listing and reading declarations do not load heavy arrays.")

@st.cache_data(show_spinner=False)
def metadata():
    return full_catalog()

records = metadata()
st.metric("Registered datasets", len(records))
suite = st.segmented_control("Collection", ["All", *SUITE_LABELS], default="All",
                              format_func=lambda x: SUITE_LABELS.get(x, x))
dimension = st.selectbox("Equation class", ["All classes", *KIND_LABELS],
                          format_func=lambda x: KIND_LABELS.get(x, x))
query = st.text_input("Search dataset or equation", placeholder="e.g. Navier–Stokes, KdV, thermal")
filtered = [
    row for row in records
    if (suite == "All" or row["suite"] == suite)
    and (dimension == "All classes" or row["kind"] == dimension)
    and (not query or query.casefold() in
         (" ".join(str(row[key]) for key in
                   ("id", "title", "source", "files"))).casefold())
]
st.caption(f"Showing {len(filtered)} of {len(records)} registered scientific datasets.")
table = pd.DataFrame([{
    "Dataset": row["id"],
    "Equation / experiment": row["title"],
    "Class": row["kind_label"],
    "Collection": row["suite_label"],
    "Reference law": "Declared" if row["law_known"] else "May be unavailable",
    "Execution": "Local EPDE" if is_local_only(row["id"]) else "Server or local (budget permitting)",
} for row in filtered])
st.dataframe(table, use_container_width=True, hide_index=True)
if filtered:
    names = [row["id"] for row in filtered]
    selected = st.selectbox("Inspect a scientific record", names,
                            format_func=lambda name: next(
                                row["title"] + " (" + name + ")" for row in filtered if row["id"] == name))
    row = next(row for row in filtered if row["id"] == selected)
    st.subheader(row["title"])
    st.write("**Dataset ID:** `" + selected + "` · **Class:** " + row["kind_label"] +
             " · **Source:** " + row["source"] + " · **Suite:** " + row["suite_label"])
    st.write("**Original registered data paths:** `" + row["files"] + "`")
    laws = declared_equations(selected)
    if laws:
        st.markdown("**Declared governing equation(s)**")
        for expression in laws:
            st.code(expression, language=None)
        st.caption("These declarations are read from the original dataset loader. "
                   "Some datasets provide only numeric data or dynamic reference laws.")
    else:
        st.info("No static governing-law literal can be extracted from this loader. "
                "This is not evidence that the method has no mathematical model.")
    if is_local_only(selected):
        st.warning("This dataset needs the full local scientific environment. "
                   "It remains visible in the catalog and can be opened in a local "
                   "copy of the same Streamlit interface; a shared free VM will not load it.")
    elif row["suite"] in ("real", "other"):
        st.info("Measured or supplementary records may not have an exact known law, "
                "and some original data files must be downloaded separately.")
    if st.button("Use this record in Lab", type="primary"):
        st.session_state["catalog_requested_record"] = selected
        st.switch_page("views/10_Discovery_lab.py")
st.caption("Scientific source: projects/pic/epde_bench/datasets.py. "
           "Datasets registered in the source are not necessarily shipped as complete "
           "data files in every checkout. See the data documentation for acquisition.")
