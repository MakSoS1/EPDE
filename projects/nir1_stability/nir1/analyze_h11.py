"""Independent-system H11 physical-law audit, no fitting or tuning.

Requires all 9*2*2 frozen full-EPDE runs from one immutable GitHub Action.
Original statuses are counted and never selectively retried or omitted.
The equal-prior postfilter is run on BASELINE Pareto fronts at zero new
evolution cost. It selects candidates truth-blind; truth is read AFTER.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .dynamic_postfilter import select_temporal_postfilter
from .metrics import _cluster_signflip_p

CASES = (
    "nir1_h11_gaussian_heat", "nir1_h11_advection_decay",
    "nir1_h11_viscous_burgers", "nir1_h11_klein_gordon",
    "nir1_h11_biharmonic", "nir1_h11_airy", "nir1_h11_shear",
    "nir1_h11_logistic_field", "nir1_h11_gauss_reactdiff",
)
METHODS = ("default", "nir1_dynamic_mdl_select")
SEEDS = (0, 1)


def score_verified(rows, *, source_sha: str):
    from projects.pic.epde_bench.datasets import load
    from projects.pic.epde_bench.metrics import canonical_tokens, hamming_best

    if len(source_sha) < 12:
        raise ValueError("Analysis requires exact source commit identity")
    expected = {(case,method,seed) for case in CASES
                for method in METHODS for seed in SEEDS}
    by_id = {}
    for row in rows:
        key = (row.get("dataset"), row.get("research_variant"), row.get("seed"))
        if key not in expected:
            raise ValueError(f"Unexpected scientific identity: {key!r}")
        if key in by_id:
            raise ValueError(f"Duplicate scientific identity: {key!r}")
        if row.get("nir1_smoke_only") is not False:
            raise ValueError("Smoke cannot substitute a full experiment")
        if row.get("config",{}).get("nir1_data_seed") != 0:
            raise ValueError("Mismatched data realization")
        env = row.get("environment",{})
        if env.get("epde_commit") and not source_sha.startswith(str(env["epde_commit"])):
            raise ValueError("Wrong EPDE source revision")
        search = row.get("search_config",{})
        evolution = search.get("evolution",{})
        if evolution.get("population_size") != 16 or evolution.get("training_epochs") != 5:
            raise ValueError(f"Budget changed in {key!r}")
        if row.get("status") == "ok":
            if len(row.get("front",())) != len(row.get("objectives",())):
                raise ValueError("Pareto front/objective vectors not aligned")
        by_id[key] = row
    if set(by_id) != expected:
        raise ValueError(f"Missing planned H11 conditions: {sorted(expected-set(by_id))!r}")

    detail = []
    for case in CASES:
        ref = [canonical_tokens(system) for system in load(case).truth_systems]
        for seed in SEEDS:
            native = by_id[(case,"default",seed)]
            h11 = by_id[(case,"nir1_dynamic_mdl_select",seed)]
            # Raw PIC configurations must be bytewise/JSON-identical
            # for both methods; variant is a scoped research flag.
            if native.get("search_config") != h11.get("search_config"):
                raise ValueError("Nonmatching PIC training/evolution configuration")
            def is_exact(row,key):
                return bool(row["metrics"].get(key)) if row.get("status") == "ok" else False
            post_idx = (select_temporal_postfilter(
                native.get("front") or [], native.get("objectives") or [])
                if native.get("status") == "ok" else None)
            post_exact = bool(hamming_best(
                canonical_tokens(native["front"][post_idx]), ref) == 0
            ) if post_idx is not None else False
            if h11.get("status") == "ok":
                from .causal_mdl_selector import select_min_causal_description
                actual = select_min_causal_description(h11["objectives"])
                if h11["metrics"].get("selected_index") != actual:
                    raise ValueError("H11 selected index does not use frozen argmin-first rule")
                if (h11.get("metrics_pic_original",{}).get("selected_index") ==
                    h11["metrics"].get("selected_index")):
                    pass  # unchanged selection is still allowed
                if h11.get("selection_strategy") != "nir1_minimum_dynamic_description_v1":
                    raise ValueError("Unrecognized H11 selection implementation")
            detail.append({
                "system":case, "optimizer_seed":seed,
                "baseline_status":native.get("status"),
                "h11_status":h11.get("status"),
                "baseline_selected_exact":is_exact(native,"success_selected"),
                "baseline_truth_front":is_exact(native,"success_front"),
                "baseline_temporal_postfilter_exact":post_exact,
                "baseline_temporal_postfilter_found":post_idx is not None,
                "h11_selected_exact":is_exact(h11,"success_selected"),
                "h11_truth_front":is_exact(h11,"success_front"),
                "h11_native_pic_selected_exact":bool(
                    h11.get("metrics_pic_original",{}).get("success_selected")
                ) if h11.get("status")=="ok" else False,
                "baseline_runtime_seconds":native.get("total_seconds"),
                "h11_runtime_seconds":h11.get("total_seconds"),
            })
    clusters = defaultdict(list)
    for row in detail:
        clusters[row["system"]].append(row)
    effects = np.asarray([np.mean([
        float(t["h11_selected_exact"]) - float(t["baseline_temporal_postfilter_exact"])
        for t in clusters[s]]) for s in CASES], dtype=float)
    baseline_effects = np.asarray([np.mean([
        float(t["h11_selected_exact"]) - float(t["baseline_selected_exact"])
        for t in clusters[s]]) for s in CASES], dtype=float)
    p_equal, _ = _cluster_signflip_p(effects, seed=11011)
    p_native, _ = _cluster_signflip_p(baseline_effects, seed=11012)
    rng = np.random.default_rng(11013)
    boot = np.asarray([100*effects[rng.integers(0,len(effects),len(effects))].mean()
                       for _ in range(5000)])
    by_system = []
    for case in CASES:
        records=clusters[case]
        by_system.append({
            "system":case, "n_pairs":len(records),
            "native_success":sum(x["baseline_selected_exact"] for x in records),
            "equal_prior_success":sum(x["baseline_temporal_postfilter_exact"] for x in records),
            "H11_success":sum(x["h11_selected_exact"] for x in records),
            "H11_truth_front":sum(x["h11_truth_front"] for x in records),
            "all_runs_ok":all(x["baseline_status"]=="ok" and x["h11_status"]=="ok"
                              for x in records),
        })
    failures=sum(r["baseline_status"]!="ok" for r in detail)+sum(
        r["h11_status"]!="ok" for r in detail)
    return {
        "status": "COMPLETE_WITH_FAILURES" if failures else "COMPLETE",
        "research_source_sha":source_sha,
        "analysis_unit":"physical system (not optimizer seed)",
        "planned_runs":36, "retrieved_runs":len(by_id), "non_ok":failures,
        "n_physical_systems":len(CASES), "n_optimizer_seeds_per_system":len(SEEDS),
        "baseline_selected_exact":sum(x["baseline_selected_exact"] for x in detail),
        "baseline_truth_front":sum(x["baseline_truth_front"] for x in detail),
        "equal_prior_postfilter_exact":sum(x["baseline_temporal_postfilter_exact"] for x in detail),
        "H11_selected_exact":sum(x["h11_selected_exact"] for x in detail),
        "H11_truth_front":sum(x["h11_truth_front"] for x in detail),
        "H11_minus_equal_prior_pp":float(100*effects.mean()),
        "H11_minus_native_pp":float(100*baseline_effects.mean()),
        "system_cluster_signflip_p_vs_equal_prior":float(p_equal),
        "system_cluster_signflip_p_vs_native":float(p_native),
        "system_cluster_bootstrap_95_ci_pp":np.quantile(boot,[.025,.975]).tolist(),
        "per_system":by_system,
        "per_run":detail,
        "limitations":[
            "Development H10 initial 8/8 result was observed before H11 rule was frozen.",
            "Nine physical laws are newly authored, not a random iid sample of nature.",
            "Prior x0=time is provided to H11; equal-prior postfilter is required.",
            "Data and 16x5 evolution budget matched; 36 full searches are scientific comparison arms.",
            "Time-level BIC-style term cost is a heuristic for correlated PDE time series.",
            "No tuning of H11 on these validation results permitted.",
            "High-order true PDEs deliberately falsify excessive complexity bias.",
            "Even if positive, new noisy/experimental systems needed for external proof.",
        ],
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--records",type=Path,required=True)
    parser.add_argument("--source-sha",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    paths=sorted(args.records.rglob("native-full.json"))
    rows=[json.loads(path.read_text(encoding="utf-8")) for path in paths]
    result=score_verified(rows,source_sha=args.source_sha)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True,
                                      allow_nan=False,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({key:result[key] for key in (
        "status","planned_runs","non_ok","H11_selected_exact",
        "equal_prior_postfilter_exact","baseline_selected_exact",
        "H11_minus_equal_prior_pp","system_cluster_signflip_p_vs_equal_prior")}))
    return 0 if result["status"]=="COMPLETE" else 2

if __name__=="__main__":
    raise SystemExit(main())
