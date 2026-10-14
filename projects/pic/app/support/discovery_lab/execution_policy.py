"""Hard resource gate for EPDE jobs on the free public Streamlit machine.

The gate uses data geometry and the *effective* search budget; it is checked
before worker execution, not merely painted as a disabled button.
Native runs explicitly bypass this gate: these limits protect ONLY hosting.
"""
from __future__ import annotations

from math import prod

# Intentionally conservative until measured on the deployed shared Streamlit VM.
MAX_CELLS = 8_000
MAX_VARIABLES = 2
MAX_POPULATION = 24
MAX_EPOCHS = 8
MAX_SEARCH_WORK = 250_000  # field cells × candidates × epochs
HEAVY_RECORDS = frozenset({
    "ns", "jhtdb", "jhtdb_plane", "jhtdb_slice", "ks", "navier_stokes",
    "kdv_homogen", "kdv_sga", "heat_laser", "dp_video",
})
# A free shared server must not materialize huge arrays just to classify a job.


def is_local_only(dataset: str | None) -> bool:
    """Classify by registry metadata, without opening numerical input data."""
    if not dataset:
        return False
    if dataset in HEAVY_RECORDS:
        return True
    from epde_bench.datasets import REGISTRY
    spec = REGISTRY.get(dataset)
    return bool(spec and spec.kind in ("pde_2d", "pde_3d"))


def classify_epde(dataset: str | None, shape, variable_count: int, search: dict):
    """Return (allowed, explanation, diagnostic metrics) for hosted EPDE.

    A coarse upper bound on search work; not a wall-clock prediction. Deny
    unknown geometry or invalid params. Native local EPDE remains unrestricted.
    """
    if is_local_only(dataset):
        return False, "This record requires local EPDE; it is not loaded on the shared free server.", {}
    try:
        dims = tuple(int(n) for n in shape)
        n_vars = int(variable_count)
        evolution = search.get("evolution", {})
        population = int(evolution.get("population_size", 16))
        epochs = int(evolution.get("training_epochs", 5))
        cells = prod(dims)
    except (ValueError, TypeError, OverflowError, AttributeError):
        return False, "Invalid geometry or search budget.", {}
    data = {"cells": cells, "variables": n_vars, "population": population,
            "epochs": epochs, "estimated_work": cells * population * epochs}
    if not dims or len(dims) > 2 or any(n < 10 for n in dims):
        return False, "Server mode supports only ODEs and small 1D spatial PDEs.", data
    if cells > MAX_CELLS or n_vars > MAX_VARIABLES:
        return False, f"Dataset exceeds hosted limits ({MAX_CELLS} grid cells, {MAX_VARIABLES} variables).", data
    if population > MAX_POPULATION or epochs > MAX_EPOCHS:
        return False, f"Budget exceeds hosted limits ({MAX_POPULATION} candidates, {MAX_EPOCHS} epochs).", data
    if population < 2 or epochs < 1:
        return False, "Invalid search budget.", data
    if cells * population * epochs > MAX_SEARCH_WORK:
        return False, "The selected input × population × epochs exceeds the free hosting budget.", data
    return True, "Suitable for a short server-side EPDE demo; execution time still varies.", data


def require_hosted_epde(dataset, shape, variable_count, search):
    allowed, reason, _ = classify_epde(dataset, shape, variable_count, search)
    if not allowed:
        raise ValueError("Hosted EPDE refused: " + reason)
