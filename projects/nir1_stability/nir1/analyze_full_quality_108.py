"""Final 108-full-search audit plus 36 no-evolution H13 diagnostics.

36 physical laws × 3 actual EPDE methods (original, H11, H12) at one
optimizer seed, full 16x5 budget. H14 is a truth-blind Pareto postselector
for the H12 population, so it adds no search. H13 is a truth-blind SVD
diagnostic, not a fourth/fifth recovery algorithm.

STRICT: require all original 72 default+H12 raw records and all 36 H11 raw
records, even failures, never silently drop. Separate previously seen 25
from H14 preregistered 11 when interpreting p-values.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

from .analyze_full_quality import (
    COHORTS, RUN_SHA, exact_signflip, run_audit
)
from .causal_mdl_selector import select_min_causal_description

H11_SHA="7fc627eadbfba9b44dc7c9549e85c474e5fca991"
H13_SHA="e346bfe6103ebc3ee478a29845e1a62821154888"

def full_108(default_and_h12, h11records, h13_data):
    original=run_audit(default_and_h12)
    if original["verified_full_searches"]!=72:
        raise ValueError("Only 72 originally planned baseline/H12 records are valid")
    mapping={r["dataset"]:r for r in original["detailed_records"]}
    if len(mapping)!=36:
        raise ValueError("Original 36 identity coverage invalid")

    expected={name for group in COHORTS.values() for name in group}
    found={}
    for row in h11records:
        key=row.get("dataset")
        if (key not in expected or key in found or
            row.get("research_variant")!="nir1_dynamic_mdl_select" or
            row.get("seed")!=0 or
            row.get("nir1_smoke_only") is not False or
            row.get("config",{}).get("nir1_data_seed")!=0):
            raise ValueError(f"Unexpected H11 identity {key}")
        cfg=row.get("search_config",{})
        if cfg.get("evolution",{}).get("population_size")!=16 or cfg.get(
            "evolution",{}).get("training_epochs")!=5:
            raise ValueError(f"H11 full budget mismatch at {key}")
        observed_sha=row.get("environment",{}).get("epde_commit")
        if observed_sha and not H11_SHA.startswith(str(observed_sha)):
            raise ValueError(f"H11 source SHA mismatch at {key}")
        if row.get("status")=="ok":
            vectors=row.get("objectives") or []
            if len(row.get("front") or [])!=len(vectors):
                raise ValueError(f"H11 Pareto front mismatch at {key}")
            frozen=select_min_causal_description(vectors)
            if row.get("metrics",{}).get("selected_index") != frozen:
                raise ValueError(f"H11 selector not frozen argmin first objective at {key}")
            if row.get("selection_strategy")!="nir1_minimum_dynamic_description_v1":
                raise ValueError(f"Unrecognized H11 implementation at {key}")
        found[key]=row

    if set(found)!=expected or len(h11records)!=36:
        raise ValueError(f"Need 36 original H11 records, missing {sorted(expected-set(found))}")

    if (h13_data.get("status")!="H13_PARALLEL_DIAGNOSTIC_NOT_EQUATION_DISCOVERY"
        or h13_data.get("observed_systems")!=36 or
        h13_data.get("EPDE_evolution_runs")!=0):
        raise ValueError("H13 result not a complete no-evolution diagnostic")
    h13records={r["dataset"]:r for r in h13_data["records"]}
    if len(h13records)!=36 or set(h13records)!=expected:
        raise ValueError("H13 diagnostic system identity mismatch")
    if any(r.get("truth_used") is not False for r in h13records.values()):
        raise ValueError("H13 accessed benchmark truth")
    detailed=[]
    for name in sorted(expected):
        p=mapping[name]
        h=found[name]
        if h.get("status")=="ok" and h.get("search_config") != (
            next(r["search_config"] for r in default_and_h12 if
                 r["dataset"]==name and r["research_variant"]=="default")):
            raise ValueError(f"Nonidentical H11 search configuration for {name}")
        h13=h13records[name]
        details={
            **p,
            "H11_status":h.get("status"),
            "H11_selected":bool(h.get("metrics",{}).get("success_selected"))
                if h.get("status")=="ok" else False,
            "H11_truth_front":bool(h.get("metrics",{}).get("success_front"))
                if h.get("status")=="ok" else False,
            "H11_error":h.get("error"),
            "H11_wall_seconds":h.get("total_seconds"),
            "H13_status":h13["status"],
            "H13_kind":h13.get("kind"),
            "H13_basis_type":h13.get("type"),
            "H13_normalized_rank_ratio":h13.get("certificate",{}).get("sigma_min_over_max"),
            "H13_diagnostic_error":h13.get("error"),
        }
        detailed.append(details)
    scores={}
    for cohort,names in COHORTS.items():
        subset=[r for r in detailed if r["cohort"]==cohort]
        assert len(subset)==len(names)
        def total(k):return sum(bool(r[k]) for r in subset)
        pairs={
            "H11_minus_original":np.asarray([
                int(r["H11_selected"])-int(r["default_selected"]) for r in subset],dtype=float),
            "H12_minus_H11":np.asarray([
                int(r["H12_selected"])-int(r["H11_selected"]) for r in subset],dtype=float),
            "H14_minus_H11":np.asarray([
                int(r["H14_closed"])-int(r["H11_selected"]) for r in subset],dtype=float),
            "H14_minus_equal_prior":np.asarray([
                int(r["H14_closed"])-int(r["baseline_equal_prior"]) for r in subset],dtype=float),
        }
        scores[cohort]={
            "n_physical_systems":len(subset),
            "original_selected":total("default_selected"),
            "H11_selected":total("H11_selected"),
            "H12_selected":total("H12_selected"),
            "H14_closed_selected":total("H14_closed"),
            "original_truth_front":total("default_truth_front"),
            "H11_truth_front":total("H11_truth_front"),
            "H12_truth_front":total("H12_truth_front"),
            "baseline_same_prior_selected":total("baseline_equal_prior"),
            "original_errors":sum(r["default_status"]!="ok" for r in subset),
            "H11_errors":sum(r["H11_status"]!="ok" for r in subset),
            "H12_errors":sum(r["H12_status"]!="ok" for r in subset),
            "H14_abstentions":total("H14_abstain"),
            "H13_diagnostics_completed":sum(r["H13_status"]=="ok" for r in subset),
            "H13_diagnostic_errors":sum(r["H13_status"]=="error" for r in subset),
            "H13_no_equation_accuracy":True,
            "paired":{k:{"delta_pp":float(100*v.mean()),
                         "two_sided_exact_signflip_p":exact_signflip(v)}
                      for k,v in pairs.items()},
        }
    return {
        "status":"COMPLETE_WITH_SEARCH_ERRORS" if any(
            r["default_status"]!="ok" or r["H11_status"]!="ok" or
            r["H12_status"]!="ok" for r in detailed) else "COMPLETE",
        "actual_EPDE_full_search_runs":108,
        "original_plus_H12":72,"new_H11":36,
        "H13_diagnostic_systems":36,
        "H13_extra_search_runs":0,"H14_extra_search_runs":0,
        "physical_systems":36,
        "same_data_and_search_budget":"one single trajectory, full 16x5 and seed=0",
        "source_commits":{"H11":H11_SHA,"H13":H13_SHA,
            "H14_original":RUN_SHA["prospective_H14"],
            "original_and_H12_retrospective":RUN_SHA["retrospective_H11"]},
        "cohorts":scores,"records":detailed,
        "limitations":[
            "H13 is a matrix-rank diagnostic, NOT a competing discovery predictor.",
            "H11 was frozen before H14 physical data were observed but its "
            "additional 36-run third-arm launch was not in the original H14 preregistration.",
            "New H14 11-system cohort and previously viewed 25-system cohorts "
            "must not be pooled for significance claims.",
            "Same optimizer seed=0 gives no across-seed variance estimate.",
            "Synthetic PDEs and their numerical derivatives are not real-world validation.",
            "Missing runs are fatal; crashes/abstentions count as incorrect.",
            "No extra data, trajectories, epochs, tuning, or E1 optimizations.",
        ],
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--prospective",type=Path,required=True)
    parser.add_argument("--retrospective",type=Path,required=True)
    parser.add_argument("--h11",type=Path,required=True)
    parser.add_argument("--h13",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    a=parser.parse_args()
    def records(p):
        return [json.loads(f.read_text(encoding="utf-8"))
                for f in sorted(p.rglob("native-full.json"))]
    rank=json.loads(a.h13.read_text(encoding="utf-8"))
    result=full_108(records(a.prospective)+records(a.retrospective),
                    records(a.h11),rank)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,
                                    sort_keys=True,allow_nan=False)+"\\n",
                        encoding="utf-8")
    print(json.dumps({"status":result["status"],
                      "evolutions":result["actual_EPDE_full_search_runs"],
                      "H13_systems":result["H13_diagnostic_systems"],
                      "cohorts":result["cohorts"]}))
if __name__=="__main__":
    main()
