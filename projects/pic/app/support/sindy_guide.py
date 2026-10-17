"""Educational guide to PySINDy as the app runs it: the PDEComp benchmark's own code.

The guide is read-only. Actual searches run only through the Lab.
"""
from __future__ import annotations

import streamlit as st
from support import diagrams
from support.common import link as page_link

STAGES = {
    "Input and derivatives": (
        "The fields and their grid are passed unchanged. With PDEComp's fixed protocol the "
        "derivatives come from the benchmark's shared NumPy finite differences; with the native "
        "protocol PySINDy computes them with its own finite differences.", "input"),
    "Candidate library": (
        "Fixed protocol: the benchmark's shared library for this dataset (powers, derivatives, "
        "coordinates, trigonometric and dataset-specific terms such as 1/x or (u u_x)_x). Native "
        "protocol: PySINDy's polynomial, trigonometric and PDE libraries. A term absent from the "
        "library cannot be recovered.", "library"),
    "Sparse regression": (
        "For the target time derivative, a sparse optimizer estimates coefficients and removes "
        "small ones. On PDEComp's datasets the optimizer and its parameters are those PDEComp set "
        "for that dataset (STLSQ or FROLS); elsewhere PDEComp's native defaults (STLSQ).", "regression"),
    "Discovered equation": (
        "The nonzero terms and their coefficients form the equation. PySINDy returns one equation "
        "per target, not a Pareto front of candidates.", "result"),
}


def _graph(selected):
    labels = {
        "input": "Grids, fields\nand derivatives",
        "library": "Candidate\nlibrary",
        "regression": "Sparse regression\nSTLSQ / FROLS",
        "result": "Selected equation",
    }
    lines = [
        "digraph sindy {", "rankdir=LR; nodesep=0.2; ranksep=0.3;",
        'node [shape=box,style="rounded,filled",fontname="Helvetica",fontsize=11];',
        'edge [color="#8796a5",arrowsize=0.65];',
    ]
    for key, label in labels.items():
        fill = "#315f9c" if key == selected else "#e8eff7"
        font = "white" if key == selected else "#1f3249"
        lines.append(f'{key} [label="{label}",fillcolor="{fill}",fontcolor="{font}"];')
    lines += ["input -> library -> regression -> result;", "}"]
    return "\n".join(lines)


def render_sindy_guide():
    st.subheader("SINDy — sparse identification of nonlinear dynamics")
    st.write(
        "SINDy searches for a small set of active terms in a predefined candidate library. "
        "Unlike the evolutionary exploration in EPDE, it uses sparse regression: "
        "most coefficients should become zero."
    )
    st.latex(r"\dot{\mathbf u}(t)\approx\Theta(\mathbf u,t)\,\Xi")
    st.caption(
        "Theta is a matrix of candidate features evaluated on data; Xi contains their "
        "coefficients. For PDEs, the features may include spatial derivatives. "
        "The target can be a first or a second time derivative."
    )

    st.markdown("### Architecture in LikeC4")
    if diagrams.available():
        st.caption(
            "Original LikeC4 diagrams from this repository: the shared benchmark runner, "
            "the preprocessing layer and the PySINDy adapter. Click the diagram to zoom in."
        )
        choice = st.selectbox("Architecture view", [
            "System landscape (includes PySINDy)",
            "Benchmark implementation (baseline adapter)",
            "Datasets, runner and application",
        ], key="sindy_architecture_view")
        view = {
            "System landscape (includes PySINDy)": "index",
            "Benchmark implementation (baseline adapter)": "benchView",
            "Datasets, runner and application": "picView",
        }[choice]
        diagrams.likec4_view(view, height=540)
    else:
        st.warning("The compiled LikeC4 bundle is unavailable; see the flow diagram below.")

    st.markdown("### Search stages")
    stage = st.selectbox("Inspect a stage", list(STAGES), key="sindy_stage")
    st.graphviz_chart(_graph(STAGES[stage][1]), use_container_width=True)
    st.write(STAGES[stage][0])

    st.markdown("### Sparse regression")
    st.latex(r"\partial_t u\approx\Theta(u,u_x,u_{xx},\ldots)\,\xi")
    st.latex(r"\min_{\xi}\ \|\partial_t u-\Theta\xi\|_2^2\quad"
             r"\text{with few nonzero entries in }\xi")
    st.caption(
        "This expresses the general identification goal. The repository's STLSQ optimizer "
        "alternates fitting and coefficient thresholding; it is not an exhaustive L0 solver."
    )

    with st.expander("Implementation details", expanded=False):
        from epde_bench import pdecomp
        st.markdown(
            f"""
**Source:** the PDEComp benchmark ([{pdecomp.PDECOMP_REPO}](https://github.com/{pdecomp.PDECOMP_REPO}),
commit {pdecomp.PDECOMP_REV[:7]}); its PySINDy runner is executed unchanged.

- **PDEComp's datasets** (the PDEComp group of records): the same data, grid and window as in
  the benchmark. *Fixed* protocol — shared derivatives and library, the optimizer tuned for the
  dataset (the setting of PDEComp's published tables); *native* — PySINDy's own differentiation
  and libraries.
- **Other data** (other records, uploads, generated ODEs): the native protocol with PDEComp's
  default settings. Only the target time derivative is given: from the left-hand side of the
  known law, or chosen for data without one.
- **Covered:** ODEs, systems of ODEs and scalar PDEs in one space dimension (Navier–Stokes only
  as PDEComp's own dataset).
- **Scoring:** the app checks the equation exactly as for EPDE (known law, HD, RE, balance and
  solution against the data); PDEComp's own HD and RE are shown next to it.
"""
        )

    st.markdown("### EPDE and SINDy — what is different?")
    st.table({
        "Aspect": [
            "Structure search", "Candidate terms", "Sparsity", "Returned candidates",
            "Fair comparison",
        ],
        "EPDE": [
            "Evolution over equations", "Sampled from token families",
            "Configured operator in candidate fitting", "Pareto level / population",
            "Identical data; EPDE's own differentiation",
        ],
        "PySINDy (PDEComp)": [
            "Regression over a fixed library", "Precomputed columns of a design matrix",
            "STLSQ or FROLS coefficient thresholding", "One equation per target",
            "Identical data; PDEComp's own differentiation",
        ],
    })
    st.info(
        "PySINDy reports no Pareto scores, so there is nothing to place next to EPDE's front. "
        "Use the Compare page: the same input, the same check against the known law and the "
        "same reconstruction metrics for both methods."
    )

    st.markdown("### Try it in the same Lab")
    st.code(
        'from epde_bench.runner import run_one\n'
        'result = run_one("ode", variant="pysindy_pdecomp_fixed", noise=0.0, seed=0)\n'
        'print(result["status"], result.get("selected"))',
        language="python",
    )
    st.caption("This is a Python example for a local checkout; the browser runs the same "
               "PySINDy through Run a search and saves the result under Results.")
    col1, col2 = st.columns(2)
    with col1:
        page_link("views/10_Discovery_lab.py", "Run PySINDy in Lab", "🧪")
    with col2:
        page_link("views/12_Discovery_compare.py", "Compare methods", "⚖️")
