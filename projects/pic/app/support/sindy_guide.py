"""Educational PySINDy guide reflecting epde_bench.baselines.run_pysindy.

The guide is read-only. Actual searches run only through the Lab.
"""
from __future__ import annotations

import streamlit as st
from support import diagrams
from support.common import link as page_link

STAGES = {
    "Input and derivatives": (
        "Coordinates and measured fields are loaded. By default SINDy uses the same "
        "EPDE derivative preprocessing and fitted-domain boundaries, including supplied "
        "derivative stacks where available.", "input"),
    "Candidate library": (
        "The program evaluates a fixed set of possible terms: variable powers, derivatives, "
        "coordinates, sine/cosine when configured, extra forcing and permitted products. "
        "A term absent from the library cannot be recovered.", "library"),
    "Sparse regression": (
        "For each admissible time derivative, PySINDy STLSQ estimates coefficients and "
        "removes coefficients below a threshold. The runner sweeps many thresholds "
        "with and without normalized columns.", "regression"),
    "Validation and selection": (
        "The model is assessed on the last 20% of interior time levels (not used in "
        "coefficient fitting). The sparsest model close to the minimum validation "
        "error is preferred. This does not establish unseen-trajectory forecasting.", "selection"),
    "Discovered equation": (
        "The nonzero terms and fitted coefficients form the selected equation system. "
        "This particular baseline returns one selected system, not a multi-point "
        "Pareto front.", "result"),
}


def _graph(selected):
    labels = {
        "input": "Grids, fields\nand derivatives",
        "library": "Fixed candidate\nlibrary",
        "regression": "Sparse regression\nSTLSQ",
        "selection": "Time-tail validation\n+ sparse selection",
        "result": "Selected equation\n+ diagnostics",
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
    lines += ["input -> library -> regression -> selection -> result;", "}"]
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
        "Our implementation can test higher time-derivative targets as well."
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
        st.markdown(
            """
**Source:** projects/pic/epde_bench/baselines.py, function run_pysindy.

- **Preprocessing:** uses the EPDE derivative estimator by default.
- **Library:** evaluated variable powers, derivatives, optional coordinate/trigonometric
  terms, forcing fields and allowed pairwise products.
- **Regressor:** PySINDy STLSQ, sweeping thresholds and normalized/unnormalized columns.
- **Selection:** validation is on the last 20% of retained time levels; choose the sparsest
  candidate within tolerance of the lowest held-out error.
- **Output:** one selected equation system; objectives are normalized validation MSE
  and active term count, rather than EPDE's configurable multiobjective values.
- **Limitations:** some temporal-derivative coefficient forms and continuity constraints
  cannot be represented by this baseline; they are marked unsupported.
"""
        )
        st.caption("These are specific properties of this project's PySINDy adapter, "
                   "not universal restrictions on all SINDy implementations.")

    st.markdown("### EPDE and SINDy — what is different?")
    st.table({
        "Aspect": [
            "Structure search", "Candidate terms", "Sparsity", "Returned candidates",
            "Fair comparison",
        ],
        "EPDE": [
            "Evolution over equations", "Sampled from token families",
            "Configured operator in candidate fitting", "Pareto level / population",
            "Identical data and preprocessing",
        ],
        "PySINDy baseline": [
            "Regression over a fixed library", "Precomputed columns of a design matrix",
            "STLSQ coefficient thresholding", "One selected equation system",
            "Identical data and preprocessing",
        ],
    })
    st.info(
        "A SINDy point with validation MSE and number of terms is not directly comparable "
        "to an EPDE Pareto point. Use the Compare page for identical-input recovery and "
        "common reconstruction metrics instead of comparing raw optimization scores."
    )

    st.markdown("### Try it in the same Lab")
    st.code(
        'from epde_bench.runner import run_one\n'
        'result = run_one("ode", variant="pysindy", noise=0.0, seed=0)\n'
        'print(result["status"], result.get("selected"))',
        language="python",
    )
    st.caption("This is a Python example for a local checkout; the browser launches "
               "the same baseline through Lab and saves the result under Results.")
    col1, col2 = st.columns(2)
    with col1:
        page_link("views/10_Discovery_lab.py", "Run PySINDy in Lab", "🧪")
    with col2:
        page_link("views/12_Discovery_compare.py", "Compare methods", "⚖️")
