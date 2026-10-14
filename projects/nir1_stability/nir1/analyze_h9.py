"""Failure-aware analysis of H9 dynamic target vs same-prior postfilter.

Works on exact JSONs extracted from the original 16 Actions artifacts.
One run per (dataset, variant, optimizer seed); do not omit unsuccessful
conditions or count optimizer seeds as physically independent systems.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

from .dynamic_postfilter import select_temporal_postfilter
from .metrics import _cluster_signflip_p


def analyze(rows: list[dict], *, systems: list[str], seeds: list[int], source_sha: str):
    from projects.pic.epde_bench.datasets import load
    from projects.pic.epde_bench.metrics import canonical_tokens, hamming_best

    expected={(system,arm,seed) for system in systems
              for arm in ("default","nir1_dynamic_anchor") for seed in seeds}
    found={}
    for row in rows:
        key=(row["dataset"],row["research_variant"],row["seed"])
        if key in found:raise ValueError(f"Duplicate H9 record: {key}")
        if key not in expected:raise ValueError(f"Unplanned H9 record: {key}")
        if row.get("nir1_smoke_only") is not False:
            raise ValueError("Smoke result cannot enter H9 scientific evaluation")
        ev=row.get("search_config",{}).get("evolution",{})
        if ev.get("population_size")!=16 or ev.get("training_epochs")!=5:
            raise ValueError("Not full matched EPDE budget")
        code=row.get("environment",{}).get("epde_commit")
        if code and not source_sha.startswith(str(code)):
            raise ValueError("Cross-commit contamination")
        found[key]=row
    if set(found)!=expected:
        raise ValueError(f"Missing {sorted(expected-set(found))}")
    score_rows=[]
    for system in systems:
        truth=[canonical_tokens(x) for x in load(system).truth_systems]
        for seed in seeds:
            baseline=found[(system,"default",seed)]
            changed=found[(system,"nir1_dynamic_anchor",seed)]
            for r in (baseline,changed):
                if r["status"]!="ok":
                    # Algorithmic errors count as failed, not dropped.
                    continue
            base_ok=baseline.get("status")=="ok"
            changed_ok=changed.get("status")=="ok"
            def metric(r,name):
                return bool(r.get("metrics",{}).get(name)) if r.get("status")=="ok" else False
            post_idx=(select_temporal_postfilter(
                baseline.get("front",[]),baseline.get("objectives",[]))
                if base_ok else None)
            post_success=(bool(hamming_best(canonical_tokens(
                baseline["front"][post_idx]),truth)==0)
                if post_idx is not None else False)
            score_rows.append({
                "system":system,"seed":seed,
                "native_status":baseline["status"],
                "dynamic_status":changed["status"],
                "native_selected":metric(baseline,"success_selected"),
                "native_front":metric(baseline,"success_front"),
                "temporal_postfilter_selected":post_success,
                "temporal_postfilter_eligible_count":(
                    sum(select_temporal_postfilter(
                        [candidate],[baseline["objectives"][i]]) is not None
                        for i,candidate in enumerate(baseline.get("front",[])))
                    if base_ok else 0),
                "dynamic_selected":metric(changed,"success_selected"),
                "dynamic_front":metric(changed,"success_front"),
                "native_wall_seconds":baseline.get("total_seconds"),
                "dynamic_wall_seconds":changed.get("total_seconds"),
            })
    bysystem=defaultdict(list)
    for r in score_rows:bysystem[r["system"]].append(r)
    effects=[]
    for system in systems:
        items=bysystem[system]
        effects.append(float(np.mean([float(v["dynamic_selected"])-
                                      float(v["temporal_postfilter_selected"])
                                      for v in items])))
    p,_=_cluster_signflip_p(np.asarray(effects),seed=20261011)
    return {
        "status":"COMPLETE_EXPLORATORY_ONLY",
        "source_sha":source_sha,"n_full_evolutions":len(expected),
        "n_physical_systems":len(systems),
        "optimizer_seeds":seeds,
        "native_selected":sum(x["native_selected"] for x in score_rows),
        "native_truth_front":sum(x["native_front"] for x in score_rows),
        "temporal_postfilter_selected":sum(x["temporal_postfilter_selected"] for x in score_rows),
        "dynamic_selected":sum(x["dynamic_selected"] for x in score_rows),
        "dynamic_truth_front":sum(x["dynamic_front"] for x in score_rows),
        "dynamic_minus_equal_prior_postfilter_pp":float(100*np.mean(effects)),
        "exact_signflip_p_by_physical_system":float(p),
        "rows":score_rows,
        "scientific_warnings":[
            "Comparison is a development study on known synthetic systems",
            "H9 adds physics prior x0=time, not ground-truth RHS or coefficient",
            "Equal-prior postfilter is applied to unchanged baseline Pareto front",
            "Optimizer seeds are repeated within physical systems",
            "n=4 independent physical systems: min 2-sided signflip p=0.125",
            "No optimization-track E1 caches enabled in H9 execution",
        ],
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--records",type=Path,required=True)
    parser.add_argument("--source-sha",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    rows=[]
    for path in sorted(args.records.rglob("native-full.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    body=analyze(rows,systems=[
        "nir1_heat","nir1_advdiff","nir1_reactiondiff","nir1_wave"],
        seeds=[0,1],source_sha=args.source_sha)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(body,indent=2,sort_keys=True,
                                      allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"dynamic":body["dynamic_selected"],
                      "postfilter":body["temporal_postfilter_selected"],
                      "baseline":body["native_selected"],
                      "p":body["exact_signflip_p_by_physical_system"]}))
if __name__=="__main__":
    main()
