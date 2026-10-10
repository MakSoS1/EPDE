"""Strict opt-in factorial EPDE variants on top of the existing PIC runner."""

from __future__ import annotations

import copy
from typing import Mapping

from .selectors import select_normalized_utopia


_VARIANTS = {
    "default": {"instability_metric": "chi2", "sparsity_cls": "vwsr"},
    "nir1_criterion_only": {"instability_metric": "chi2", "sparsity_cls": "vwsr",
                            "research_objective_metric": "nir1_excess"},
    "nir1_regulator_only": {"instability_metric": "chi2", "sparsity_cls": "nir1_adaptive",
                            "research_regularizer_metric": "nir1_excess"},
    "nir1_combined": {"instability_metric": "chi2", "sparsity_cls": "nir1_adaptive",
                      "research_objective_metric": "nir1_excess",
                      "research_regularizer_metric": "nir1_excess"},
}


def resolve_research_variant(name: str) -> dict[str, object]:
    if name not in _VARIANTS:
        raise ValueError(f"Unsupported full-search variant {name!r}; historical proxy is S0-only")
    return copy.deepcopy(_VARIANTS[name])


def run_nir1_epde(dataset: str, variant: str, seed: int, noise: float = 0.,
                  *, data_seed: int | None = None,
                  overrides: Mapping[str, object] | None = None) -> dict[str, object]:
    """Execute actual PIC EPDE evolution, preserving native records and failures.

    This function may consume substantial CPU; use a fresh subprocess per
    identity and evaluate one S1 pilot before scheduling confirmation grids.
    """
    from projects.pic.epde_bench.config import deep_merge
    from projects.pic.epde_bench.runner import run_one

    objectives = resolve_research_variant(variant)
    base = {"search": {"objectives": objectives}}
    effective = deep_merge(base, dict(overrides or {}))
    if data_seed is not None:
        effective["nir1_data_seed"] = int(data_seed)
    record = run_one(dataset, "default", noise, seed, overrides=effective)
    record["research_variant"] = variant
    record["research_overrides"] = effective
    if record.get("status") == "ok":
        chosen = select_normalized_utopia(record.get("objectives") or [])
        record["research_selected_index"] = chosen
        record["research_selected"] = (record["front"][chosen] if chosen is not None else None)
        # Truth can be used later for offline scoring, never in selection.
    return record
