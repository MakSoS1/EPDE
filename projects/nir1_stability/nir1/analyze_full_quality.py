"""NIR1 complete 72-run EPDE quality audit.

Entirety: 11 new prospective PDE x two search arms (GitHub Actions H14);
25 already-viewed physical systems x same arms (full-retrospective).
Every original full EPDE result is included even if it errored; zero extra
trajectory, seed, population, or epochs. H14 is an OFFLINE truth-free choice
from H12 Pareto fronts; equal-prior baseline is the same choice from native
baseline fronts. Only AFTER all indices are locked is benchmark truth read.

DO NOT pool prospective 11 with retrospective 25 for confirmatory p-values.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from .closed_evolution_selector import choose_causal_closed_front
from .analyze_h14 import CASES as PROSPECTIVE

S1=("ac","burgers","duffing","lv","ode","vdp")
S2E=("nir1_decay","nir1_harmonic","nir1_logistic","nir1_quadratic",
     "nir1_cubic","nir1_advection","nir1_heat","nir1_reactiondiff",
     "nir1_wave","nir1_advdiff")
H11=("nir1_h11_gaussian_heat","nir1_h11_advection_decay",
     "nir1_h11_viscous_burgers","nir1_h11_klein_gordon",
     "nir1_h11_biharmonic","nir1_h11_airy","nir1_h11_shear",
     "nir1_h11_logistic_field","nir1_h11_gauss_reactdiff")
COHORTS={"prospective_H14":tuple(PROSPECTIVE),
         "retrospective_S1":S1,"retrospective_S2e":S2E,
         "retrospective_H11":H11}
RUN_SHA={
    "prospective_H14":"e8aea98741016cc5b86f217205738aa486066074",
    "retrospective_S1":"1183d384697dee7b38a2f3418501160b2b480266",
    "retrospective_S2e":"1183d384697dee7b38a2f3418501160b2b480266",
    "retrospective_H11":"1183d384697dee7b38a2f3418501160b2b480266",
}
METHODS=("default","nir1_dynamic_mdl_repair")
EXPECTED={(case,method,0) for cases in COHORTS.values()
          for case in cases for method in METHODS}
FIELDS=("default_selected","baseline_equal_prior","H12_selected","H14_closed")


def exact_signflip(deltas):
    x=np.asarray(deltas,dtype=float)
    x=x[np.abs(x)>1e-12]
    if x.size==0:return 1.
    obs=abs(float(x.sum()))
    hits=sum(abs(sum(v*s for v,s in zip(x,sign)))>=obs-1e-12
             for sign in itertools.product((-1.,1.),repeat=len(x)))
    return float(hits/2**len(x))


def run_audit(records):
    from projects.pic.epde_bench.datasets import load
    from projects.pic.epde_bench.metrics import canonical_tokens,hamming_best

    lookup={}
    cohort_by_case={case:cohort for cohort,cases in COHORTS.items()
                    for case in cases}
    assert len(cohort_by_case)==36 and len(EXPECTED)==72
    for row in records:
        key=row.get("dataset"),row.get("research_variant"),row.get("seed")
        if key not in EXPECTED:raise ValueError(f"Unplanned identity {key}")
        if key in lookup:raise ValueError(f"Duplicate run {key}")
        if row.get("nir1_smoke_only") is not False:
            raise ValueError(f"Smoke in full benchmark {key}")
        evolution=row.get("search_config",{}).get("evolution",{})
        if evolution.get("population_size")!=16 or evolution.get("training_epochs")!=5:
            raise ValueError(f"Changed full-search budget {key}: {evolution}")
        if row.get("config",{}).get("nir1_data_seed")!=0:
            raise ValueError(f"Physical data seed altered {key}")
        if row.get("status")=="ok" and len(row.get("front",()))!=len(row.get("objectives",())):
            raise ValueError(f"Corrupted front/objective alignment {key}")
        expected_sha=RUN_SHA[cohort_by_case[key[0]]]
        observed_sha=row.get("environment",{}).get("epde_commit")
        if observed_sha and not expected_sha.startswith(str(observed_sha)):
            raise ValueError(f"Unexpected source revision in {key}: {observed_sha}")
        lookup[key]=row
    if set(lookup)!=EXPECTED:
        raise ValueError(f"MISSING {len(EXPECTED-set(lookup))} original full records: "
                         f"{sorted(EXPECTED-set(lookup))}")

    # TRUTH-BLIND choice indices, before loading *any* dataset truth.
    selected={}
    for case in cohort_by_case:
        for arm in METHODS:
            r=lookup[(case,arm,0)]
            selected[(case,arm)]=(choose_causal_closed_front(
                r.get("front") or [],r.get("objectives") or [])
                if r.get("status")=="ok" else None)

    rows=[]
    for cohort,cases in COHORTS.items():
        for case in cases:
            a=lookup[(case,"default",0)]
            b=lookup[(case,"nir1_dynamic_mdl_repair",0)]
            if a.get("search_config")!=b.get("search_config"):
                raise ValueError(f"Search config differs across methods: {case}")
            truth=[canonical_tokens(x) for x in load(case).truth_systems]

            def eligible_correct(r,which):
                index=selected[(case,which)]
                return bool(hamming_best(
                    canonical_tokens(r["front"][index]), truth)==0
                ) if index is not None and r.get("status")=="ok" else False

            def metric(r,metric_name):
                return (bool(r.get("metrics",{}).get(metric_name))
                        if r.get("status")=="ok" else False)

            rows.append({
                "cohort":cohort,"dataset":case,"seed":0,
                "default_status":a.get("status"),
                "H12_status":b.get("status"),
                "default_selected":metric(a,"success_selected"),
                "default_truth_front":metric(a,"success_front"),
                "baseline_equal_prior":eligible_correct(a,"default"),
                "H12_selected":metric(b,"success_selected"),
                "H12_truth_front":metric(b,"success_front"),
                "H14_closed":eligible_correct(b,"nir1_dynamic_mdl_repair"),
                "baseline_prior_abstain":selected[(case,"default")] is None,
                "H14_abstain":selected[(case,"nir1_dynamic_mdl_repair")] is None,
                "baseline_error":a.get("error"),
                "H12_error":b.get("error"),
                "baseline_wall_seconds":a.get("total_seconds"),
                "H12_wall_seconds":b.get("total_seconds"),
            })
    per_cohort={}
    for cohort,cases in COHORTS.items():
        subset=[row for row in rows if row["cohort"]==cohort]
        deltas=np.asarray([
            float(row["H14_closed"])-float(row["baseline_equal_prior"])
            for row in subset],dtype=float)
        naive=np.asarray([
            float(row["H14_closed"])-float(row["default_selected"])
            for row in subset],dtype=float)
        post=np.asarray([
            float(row["H14_closed"])-float(row["H12_selected"])
            for row in subset],dtype=float)
        rng=np.random.default_rng(20261011)
        boot=np.asarray([100*float(np.mean(
            deltas[rng.integers(0,len(deltas),len(deltas))]))
            for _ in range(10000)])
        per_cohort[cohort]={
            "n_physical_systems":len(subset),
            **{key:sum(int(row[key]) for row in subset) for key in (
                "default_selected","baseline_equal_prior","H12_selected",
                "H14_closed","default_truth_front","H12_truth_front",
                "baseline_prior_abstain","H14_abstain")},
            "default_errors":sum(row["default_status"]!="ok" for row in subset),
            "H12_errors":sum(row["H12_status"]!="ok" for row in subset),
            "H14_minus_equal_prior_pp":float(deltas.mean()*100),
            "H14_minus_native_pp":float(naive.mean()*100),
            "H14_minus_H12_pp":float(post.mean()*100),
            "exact_signflip_p_vs_equal_prior":exact_signflip(deltas),
            "exact_signflip_p_vs_native":exact_signflip(naive),
            "exact_signflip_p_vs_H12":exact_signflip(post),
            "bootstrap95_H14_minus_equal_prior_pp":np.quantile(
                boot,[.025,.975]).tolist(),
        }

    return {
        "status":"COMPLETE_WITH_ERRORS" if any(
            row["default_status"]!="ok" or row["H12_status"]!="ok"
            for row in rows) else "COMPLETE",
        "original_actions_runs":{
            "prospective_H14":38137822492,"retrospective_25":38139042448},
        "original_source_shas":{
            "prospective_H14":RUN_SHA["prospective_H14"],
            "retrospective_25":RUN_SHA["retrospective_H11"]},
        "verified_full_searches":len(lookup),
        "physical_systems":len(cohort_by_case),
        "identical_evolution_budget":"16 population * 5 epochs, 1 trajectory, seed=0",
        "primary_confirmatory_cohort":"prospective_H14",
        "cohorts":per_cohort,
        "detailed_records":rows,
        "scope_warnings":[
            "Only the 11 prospective H14 systems can support a preregistered "
            "within-study test. Remaining 25 systems were previously viewed.",
            "Some synthetic PDEs are highly related; independent physical "
            "systems is a working cluster approximation, not random nature.",
            "H14 filters already-searched H12 fronts; NO additional search.",
            "H14 prior excludes true damped dynamics with extra time "
            "derivative. That counterexample is retained.",
            "A crash, timeout or H14 abstention counts as incorrect.",
            "No noise or independent whole-trajectory test in this benchmark.",
            "One optimizer seed cannot quantify across-seed variance.",
            "No E1 performance optimizations are enabled.",
        ],
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--h14-original",required=True,type=Path)
    parser.add_argument("--retrospective-original",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    p=parser.parse_args()
    paths=(list(sorted(p.h14_original.rglob("native-full.json"))) +
           list(sorted(p.retrospective_original.rglob("native-full.json"))))
    result=run_audit([json.loads(path.read_text(encoding="utf-8"))
                      for path in paths])
    p.output.parent.mkdir(parents=True,exist_ok=True)
    p.output.write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,
                                   indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"verified_full_searches":result["verified_full_searches"],
                      "status":result["status"],
                      "prospective":result["cohorts"]["prospective_H14"]}))
if __name__=="__main__":
    main()
