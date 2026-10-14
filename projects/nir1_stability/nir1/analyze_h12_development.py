"""Audit H12 terminal target repair against frozen H11, not fresh confirmation.

All 18 H12 full 16x5 runs (9 PDE, paired optimizer seeds 0/1) must be
present. Compare to immutable 18 H11 observations from 36-run validation.
Do not reclassify H11 failures as missing. H12 was designed after seeing
those failures, so all effects are development only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

from .analyze_h11 import CASES, SEEDS
from .metrics import _cluster_signflip_p


def evaluate(h12, original, *, source_sha):
    expected={(system,seed) for system in CASES for seed in SEEDS}
    old={}
    new={}
    for row in original:
        if row.get("research_variant") not in {"default","nir1_dynamic_mdl_select"}:
            continue
        key=(row["dataset"],row["seed"],row["research_variant"])
        if key in old:raise ValueError(f"Duplicate H11 original {key}")
        old[key]=row
    for row in h12:
        key=(row["dataset"],row["seed"])
        if key not in expected:raise ValueError(f"Unknown H12 condition {key}")
        if key in new:raise ValueError(f"Duplicate H12 condition {key}")
        if row.get("research_variant")!="nir1_dynamic_mdl_repair":
            raise ValueError("Unexpected research arm")
        if row.get("nir1_smoke_only") is not False:
            raise ValueError("H12 smoke included in full analysis")
        if row.get("config",{}).get("nir1_data_seed")!=0:
            raise ValueError("Changed physical data seed")
        if row.get("search_config",{}).get("evolution",{}) != {
            "population_size":16,"training_epochs":5
        }:
            # Reject changed settings, including any silently increased
            # epochs, populations or restarts.
            raise ValueError("Changed EPDE search budget")
        if not source_sha.startswith(str(row.get("environment",{}).get("epde_commit",""))):
            raise ValueError("Wrong source SHA")
        new[key]=row
    if set(new)!=expected:
        raise ValueError(f"Missing original H12 full trials: {sorted(expected-set(new))}")
    if len(old)!=len(CASES)*len(SEEDS)*2:
        raise ValueError("Missing original frozen 36 H11+baseline records")
    table=[]
    for case in CASES:
        for seed in SEEDS:
            base=old[(case,seed,"default")]
            h11=old[(case,seed,"nir1_dynamic_mdl_select")]
            h12=new[(case,seed)]
            if not (base.get("search_config")==h11.get("search_config")==h12.get("search_config")):
                raise ValueError("Baseline/H11/H12 config mismatch")
            def flag(r,m):
                return bool(r.get("metrics",{}).get(m)) if r.get("status")=="ok" else False
            table.append({
                "system":case,"seed":seed,
                "baseline":flag(base,"success_selected"),
                "H11":flag(h11,"success_selected"),
                "H12":flag(h12,"success_selected"),
                "H11_truth_front":flag(h11,"success_front"),
                "H12_truth_front":flag(h12,"success_front"),
                "H11_status":h11.get("status"),"H12_status":h12.get("status"),
                "H11_time_s":h11.get("total_seconds"),
                "H12_time_s":h12.get("total_seconds"),
                "H12_error":h12.get("error"),
            })
    effects=[]
    by_case=[]
    for c in CASES:
        v=[row for row in table if row["system"]==c]
        eff=float(np.mean([int(z["H12"])-int(z["H11"]) for z in v]))
        effects.append(eff)
        by_case.append({
            "case":c, "H11_correct":sum(x["H11"] for x in v),
            "H12_correct":sum(x["H12"] for x in v),
            "H11_errors":sum(x["H11_status"]!="ok" for x in v),
            "H12_errors":sum(x["H12_status"]!="ok" for x in v),
            "paired_effect":eff,
        })
    p,_=_cluster_signflip_p(np.asarray(effects,dtype=float),seed=20261011)
    return {
        "status":"DEVELOPMENT_COMPLETE_WITH_FAILURES"
        if any(x["H12_status"]!="ok" for x in table) else "DEVELOPMENT_COMPLETE",
        "source_sha_H11_original":"eaae9063e625c93a43dc724b2070c0e4b3c9bb4d",
        "source_sha_H12":source_sha,"n_systems":len(CASES),
        "full_H11_baseline_trials":len(old),"full_H12_trials":len(new),
        "H11_correct":sum(x["H11"] for x in table),
        "H12_correct":sum(x["H12"] for x in table),
        "H11_errors":sum(x["H11_status"]!="ok" for x in table),
        "H12_errors":sum(x["H12_status"]!="ok" for x in table),
        "H12_minus_H11_pp":float(np.mean(effects)*100.),
        "cluster_signflip_p":float(p),"per_system":by_case,"paired_rows":table,
        "limitation":"H12 was created after inspecting H11 on these same 9 laws. "
                     "All results are exploratory; no confirmatory H12 claim.",
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--h11",type=Path,required=True)
    p.add_argument("--h12",type=Path,required=True)
    p.add_argument("--source-sha",required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    def read(folder,glob):
        return [json.loads(x.read_text(encoding="utf-8"))
                for x in sorted(folder.rglob(glob))]
    result=evaluate(read(a.h12,"native-full.json"),
                    read(a.h11,"native-full.json"),
                    source_sha=a.source_sha)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,
                                   sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({k:result[k] for k in ["status","H11_correct","H12_correct",
                                          "H11_errors","H12_errors","H12_minus_H11_pp",
                                          "cluster_signflip_p"]}))
if __name__=="__main__":
    main()
