"""Strict opt-in factorial EPDE variants on top of the existing PIC runner."""

from __future__ import annotations

import copy
import os
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
    # Conditional deletion-evidence regularizer; a development hypothesis.
    "nir1_conditional_regulator": {"instability_metric": "chi2",
                                   "sparsity_cls": "nir1_adaptive",
                                   "research_regularizer_metric": "nir1_conditional"},
    # Separate, preregistered regularizer rescue hypothesis (no selector change).
    "nir1_protected_regulator": {"instability_metric": "chi2",
                                 "sparsity_cls": "nir1_adaptive",
                                 "research_regularizer_metric": "nir1_protected"},
    # Novel H9: same original EPDE library, VWSR, 16×5 budget, ONE IC.
    # Evolution-law assumption x0=time is an EXPLICIT prior; no coefficients
    # or ground-truth supports are consulted during target selection.
    "nir1_dynamic_anchor": {"instability_metric": "chi2", "sparsity_cls": "vwsr"},
    "nir1_dynamic_mdl": {"instability_metric": "chi2", "sparsity_cls": "vwsr"},
    "nir1_dynamic_mdl_select": {"instability_metric": "chi2", "sparsity_cls": "vwsr"},
    # H12 changes ONLY capped temporal target recovery; H11 remains immutable.
    "nir1_dynamic_mdl_repair": {"instability_metric": "chi2", "sparsity_cls": "vwsr"},
    "nir1_dynamic_mdl_closed": {"instability_metric": "chi2", "sparsity_cls": "vwsr"},
    "nir1_dynamic_postfilter": {"instability_metric": "chi2", "sparsity_cls": "vwsr"},
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
    # EPDE uses one process-global operator pipeline. Research override is
    # scoped to this call and restored even on failures; CI executes each
    # method in an independent process to prevent state/seed contamination.
    previous_dynamic = os.environ.get("NIR1_DYNAMIC_TARGET")
    previous_mdl = os.environ.get("NIR1_DYNAMIC_MDL")
    previous_repair = os.environ.get("NIR1_DYNAMIC_REPAIR")
    try:
        if variant in {"nir1_dynamic_anchor", "nir1_dynamic_mdl", "nir1_dynamic_mdl_select", "nir1_dynamic_mdl_repair", "nir1_dynamic_mdl_closed"}:
            os.environ["NIR1_DYNAMIC_TARGET"] = "1"
        else:
            os.environ.pop("NIR1_DYNAMIC_TARGET", None)
        if variant in {"nir1_dynamic_mdl", "nir1_dynamic_mdl_select", "nir1_dynamic_mdl_repair", "nir1_dynamic_mdl_closed"}:
            os.environ["NIR1_DYNAMIC_MDL"] = "1"
        else:
            os.environ.pop("NIR1_DYNAMIC_MDL", None)
        if variant in {"nir1_dynamic_mdl_repair", "nir1_dynamic_mdl_closed"}:
            os.environ["NIR1_DYNAMIC_REPAIR"] = "1"
        else:
            os.environ.pop("NIR1_DYNAMIC_REPAIR", None)
        record = run_one(dataset, "default", noise, seed, overrides=effective)
    finally:
        for name, previous in (
            ("NIR1_DYNAMIC_TARGET", previous_dynamic),
            ("NIR1_DYNAMIC_MDL", previous_mdl),
            ("NIR1_DYNAMIC_REPAIR", previous_repair),
        ):
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous
    record["research_variant"] = variant
    if variant == "nir1_dynamic_anchor":
        record["research_physical_prior"] = (
            "Evolution equation: bare temporal derivative of explained variable "
            "is required as implicit RPS target; x0 is time. No true RHS, "
            "coefficients, or true derivative order is supplied.")
    if variant in {"nir1_dynamic_mdl", "nir1_dynamic_mdl_select"}:
        record["research_physical_prior"] = (
            "Same time-derivative-only target as H9 PLUS data-budget-aware BIC "
            "penalty on active RHS terms and excess high derivatives, evaluated "
            "during every RPS comparison and the MOEA/D fitness. No truth used.")
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
        if variant == "nir1_dynamic_postfilter":
            # Baseline evolution is unchanged; only apply the very same
            # temporal-law requirement after the regular Pareto search.
            from .dynamic_postfilter import select_temporal_postfilter
            from projects.pic.epde_bench.datasets import load
            from projects.pic.epde_bench.metrics import canonical_tokens, hamming_best
            chosen = select_temporal_postfilter(
                record.get("front") or [], record.get("objectives") or [])
            record["metrics_pic_original"] = copy.deepcopy(record.get("metrics", {}))
            record["research_selected_index"] = chosen
            record["research_selected"] = (record["front"][chosen] if chosen is not None else None)
            record["selected"] = record["research_selected"]
            score = record["metrics"]
            score["selected_index"] = chosen
            truth_systems = load(dataset, **record.get("config", {}).get("loader", {})).truth_systems
            if truth_systems:
                truth = [canonical_tokens(s) for s in truth_systems]
                hd = hamming_best(canonical_tokens(record["front"][chosen]), truth) if chosen is not None else None
                score["success_selected"] = bool(hd == 0) if hd is not None else False
                score["hamming_selected"] = hd
            record["selection_strategy"] = "nir1_dynamic_temporal_postfilter"
            record["research_physical_prior"] = (
                "Evolution-law target must contain bare time derivative, applied "
                "only after the original baseline has built its Pareto front.")
        if variant in {"nir1_dynamic_mdl_select", "nir1_dynamic_mdl_repair",
                       "nir1_dynamic_mdl_closed"}:
            from .causal_mdl_selector import select_min_causal_description
            from .closed_evolution_selector import choose_causal_closed_front
            # Selection is COMPLETED before loading simulator truth.
            selected = (choose_causal_closed_front(
                            record.get("front") or [], record.get("objectives") or [])
                        if variant == "nir1_dynamic_mdl_closed" else
                        select_min_causal_description(record.get("objectives") or []))
            record["metrics_pic_original"] = copy.deepcopy(record.get("metrics", {}))
            record["research_selected_index"] = selected
            record["research_selected"] = (record["front"][selected]
                                           if selected is not None else None)
            record["selected"] = record["research_selected"]
            score = record["metrics"]
            score["selected_index"] = selected
            from projects.pic.epde_bench.datasets import load
            from projects.pic.epde_bench.metrics import canonical_tokens, hamming_best
            accepted = load(dataset, **record.get("config", {}).get("loader", {})).truth_systems
            if accepted:
                truth = [canonical_tokens(s) for s in accepted]
                dist = (hamming_best(canonical_tokens(record["front"][selected]), truth)
                        if selected is not None else None)
                score["success_selected"] = bool(dist == 0) if dist is not None else False
                score["hamming_selected"] = dist
            record["selection_strategy"] = (
                "nir1_h14_causal_normal_form_v1"
                if variant == "nir1_dynamic_mdl_closed" else
                "nir1_minimum_dynamic_description_h12_repair"
                if variant == "nir1_dynamic_mdl_repair"
                else "nir1_minimum_dynamic_description_v1")
            if variant == "nir1_dynamic_mdl_closed":
                record["research_physical_prior"] = (
                    "Explicit causal-normal-form evolution: exactly one bare "
                    "temporal derivative can be isolated on EITHER side; all "
                    "other terms lack time derivatives. Damped equations with "
                    "RHS u_t are excluded; if none fit, abstain and count error.")
        # Ground truth is accessed ONLY after a fixed truth-free selection.
    return record
