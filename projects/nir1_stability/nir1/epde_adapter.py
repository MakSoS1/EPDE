"""Strict opt-in factorial EPDE variants on top of the existing PIC runner."""

from __future__ import annotations

import copy
from typing import Mapping

from .selectors import select_normalized_utopia, select_sparsefront


_VARIANTS = {
    "default": {"instability_metric": "chi2", "sparsity_cls": "vwsr"},
    "nir1_criterion_only": {"instability_metric": "chi2", "sparsity_cls": "vwsr",
                            "research_objective_metric": "nir1_excess"},
    "nir1_regulator_only": {"instability_metric": "chi2", "sparsity_cls": "nir1_adaptive",
                            "research_regularizer_metric": "nir1_excess"},
    "nir1_combined": {"instability_metric": "chi2", "sparsity_cls": "nir1_adaptive",
                      "research_objective_metric": "nir1_excess",
                      "research_regularizer_metric": "nir1_excess"},
    # Separate, preregistered regularizer rescue hypothesis (no selector change).
    "nir1_protected_regulator": {"instability_metric": "chi2",
                                 "sparsity_cls": "nir1_adaptive",
                                 "research_regularizer_metric": "nir1_protected"},
    # Baseline EPDE evolution/regularization, new truth-free front selector ONLY.
    # Frozen after S1 development; evaluation MUST use distinct systems.
    "nir1_sparsefront": {"instability_metric": "chi2", "sparsity_cls": "vwsr"},
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
        chosen = (select_sparsefront(record.get("front") or [], record.get("objectives") or [])
                  if variant == "nir1_sparsefront" else
                  select_normalized_utopia(record.get("objectives") or []))
        record["research_selected_index"] = chosen
        record["research_selected"] = (record["front"][chosen] if chosen is not None else None)
        # Keep the original PIC metrics and verdict for audit. The new selector
        # is frozen before scoring, so the later metric computation cannot leak
        # simulator truth into candidate selection.
        if variant == "nir1_sparsefront":
            from projects.pic.epde_bench.datasets import load
            from projects.pic.epde_bench.metrics import canonical_tokens, hamming_best
            record["metrics_pic_original"] = copy.deepcopy(record.get("metrics", {}))
            truth_systems = load(dataset, **record.get("config", {}).get("loader", {})).truth_systems
            record["selected"] = record["research_selected"]
            score = record["metrics"]
            score["selected_index"] = chosen
            if not truth_systems or chosen is None:
                score["success_selected"] = False if truth_systems else None
                score["hamming_selected"] = None
            else:
                accepted = [canonical_tokens(sys_) for sys_ in truth_systems]
                distance = hamming_best(canonical_tokens(record["front"][chosen]), accepted)
                score["success_selected"] = bool(distance == 0)
                score["hamming_selected"] = int(distance)
            record["selection_strategy"] = "nir1_sparsefront_v1_s1_tuned"
        # Ground truth is accessed ONLY after a fixed truth-free selection.
    return record
