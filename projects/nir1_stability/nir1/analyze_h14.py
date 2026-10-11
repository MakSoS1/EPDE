"""H14: preregistered causal-closure audit over 11 new physical PDEs.

One full EPDE run per (physical dataset, base/H12), each 16x5, no restarts.
H14 and equal-prior baseline postfilters reuse EXACT original Pareto fronts
with zero new evolution. Truth is read ONLY AFTER both choices were made.

No hyperparameters, penalties, or thresholds are fitted in this script.
Abort on missing experiments; algorithmic failures count as inaccurate.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from .closed_evolution_selector import choose_causal_closed_front
from .causal_mdl_selector import select_min_causal_description

CASES=(
    "nir1_h14_damped_wave",
    "nir1_h14_cole_hopf",
    "nir1_h14_kpp_front",
    "nir1_h14_linear_ks",
    "nir1_h14_beam",
    "nir1_h14_airy_transport",
    "nir1_h14_cubic_saturation",
    "nir1_h14_driven_advection",
    "nir1_h14_variable_diffusivity",
    "nir1_h14_liouville_growth",
    "nir1_h14_hyperdiffusion_growth",
)
SEED=0
METHODS=("default","nir1_dynamic_mdl_repair")


def exact_two_sided_signflip(deltas):
    signed=np.asarray(deltas,dtype=float)
    positive=signed[np.abs(signed)>1e-12]
    if len(positive)==0:
        return 1.0
    observed=abs(float(positive.sum()))
    outcomes=(sum(abs(sum(s*v for s,v in zip(signs,positive)))>=observed-1e-12
                  for signs in itertools.product((-1.,1.),repeat=len(positive))))
    return float(outcomes/(2**len(positive)))


def analyze(records, *, research_sha):
    from projects.pic.epde_bench.datasets import load
    from projects.pic.epde_bench.metrics import canonical_tokens, hamming_best

    expected={(case,method,SEED) for case in CASES for method in METHODS}
    by_key={}
    for row in records:
        key=(row.get("dataset"),row.get("research_variant"),row.get("seed"))
        if key not in expected:
            raise ValueError(f"Unexpected H14 source record {key}")
        if key in by_key:
            raise ValueError(f"Duplicate H14 source record {key}")
        if row.get("nir1_smoke_only") is not False:
            raise ValueError(f"Smoke is not a scientific result {key}")
        config=row.get("search_config",{})
        if config.get("evolution",{}).get("population_size")!=16 or config.get(
            "evolution",{}).get("training_epochs")!=5:
            raise ValueError(f"Mismatched evolutionary budget {key}")
        if row.get("config",{}).get("nir1_data_seed")!=0:
            raise ValueError(f"Physical data seed changed {key}")
        if row.get("status")=="ok" and (
            len(row.get("front",[]))!=len(row.get("objectives",[]))):
            raise ValueError(f"Corrupt Pareto objective alignment {key}")
        source=row.get("environment",{}).get("epde_commit")
        if source and not research_sha.startswith(str(source)):
            raise ValueError(f"Wrong source revision {key}: {source}")
        by_key[key]=row
    if set(by_key)!=expected:
        raise ValueError(f"INCOMPLETE 22 identities; missing {sorted(expected-set(by_key))}")

    # All truth-free choices are frozen BEFORE the benchmark truth is loaded.
    decisions={}
    for case in CASES:
        baseline=by_key[(case,"default",SEED)]
        h12=by_key[(case,"nir1_dynamic_mdl_repair",SEED)]
        if baseline.get("search_config")!=h12.get("search_config"):
            raise ValueError(f"Different search budget/config at {case}")
        for arm,r in (("default",baseline),("H12",h12)):
            if r.get("status")=="ok":
                front=r["front"]
                objectives=r["objectives"]
                decisions[(case,arm,"causal")] = choose_causal_closed_front(
                    front, objectives)
                decisions[(case,arm,"MDL")] = select_min_causal_description(objectives)
            else:
                decisions[(case,arm,"causal")]=None
                decisions[(case,arm,"MDL")]=None

    rows=[]
    for case in CASES:
        b=by_key[(case,"default",SEED)]
        h=by_key[(case,"nir1_dynamic_mdl_repair",SEED)]
        truth=[canonical_tokens(s) for s in load(case).truth_systems]
        def correct(r,index):
            if r.get("status")!="ok" or index is None:
                return False
            return bool(hamming_best(canonical_tokens(r["front"][index]),truth)==0)
        def original_correct(r):
            return bool(r.get("metrics",{}).get("success_selected")) if r.get("status")=="ok" else False

        bi=decisions[(case,"default","causal")]
        hi=decisions[(case,"H12","causal")]
        native=bool(original_correct(b))
        base_prior=correct(b,bi)
        h12=bool(original_correct(h))
        h14=correct(h,hi)
        rows.append({
            "dataset":case,
            "seed":SEED,
            "baseline_status":b.get("status"),
            "H12_status":h.get("status"),
            "baseline_selected_exact":native,
            "baseline_truth_on_front":bool(b.get("metrics",{}).get("success_front")) if b.get("status")=="ok" else False,
            "baseline_equal_prior_closed_exact":base_prior,
            "baseline_equal_prior_closed_available":bi is not None,
            "H12_selected_exact":h12,
            "H12_truth_on_front":bool(h.get("metrics",{}).get("success_front")) if h.get("status")=="ok" else False,
            "H14_closed_exact":h14,
            "H14_closed_available":hi is not None,
            "H14_closed_index":hi,
            "H12_native_index":h.get("metrics",{}).get("selected_index"),
            "baseline_time_seconds":b.get("total_seconds"),
            "H12_time_seconds":h.get("total_seconds"),
            "baseline_error":b.get("error"),
            "H12_error":h.get("error"),
        })
    def total(field):
        return sum(int(bool(row[field])) for row in rows)
    differences=np.asarray([
        float(row["H14_closed_exact"])-float(row["baseline_equal_prior_closed_exact"])
        for row in rows],dtype=float)
    rawdiff=np.asarray([
        float(row["H14_closed_exact"])-float(row["baseline_selected_exact"])
        for row in rows],dtype=float)
    alt=np.asarray([
        float(row["H14_closed_exact"])-float(row["H12_selected_exact"])
        for row in rows],dtype=float)
    rng=np.random.default_rng(14011)
    cluster_boot=[100*float(np.mean(differences[
        rng.integers(0,len(differences),size=len(differences))]))
        for _ in range(5000)]
    return {
        "status":"COMPLETE",
        "source_sha":research_sha,
        "n_physical_systems":len(CASES),
        "planned_full_searches":len(expected),
        "verified_full_searches":len(by_key),
        "failed_baseline":sum(row["baseline_status"]!="ok" for row in rows),
        "failed_H12":sum(row["H12_status"]!="ok" for row in rows),
        "baseline_native_exact":total("baseline_selected_exact"),
        "baseline_equal_prior_exact":total("baseline_equal_prior_closed_exact"),
        "H12_native_exact":total("H12_selected_exact"),
        "H14_closed_exact":total("H14_closed_exact"),
        "baseline_truth_on_front":total("baseline_truth_on_front"),
        "H12_truth_on_front":total("H12_truth_on_front"),
        "H14_abstentions":sum(row["H14_closed_available"] is False for row in rows),
        "baseline_prior_abstentions":sum(row["baseline_equal_prior_closed_available"] is False for row in rows),
        "H14_minus_equal_prior_pp":float(100*np.mean(differences)),
        "H14_minus_baseline_pp":float(100*np.mean(rawdiff)),
        "H14_minus_H12_pp":float(100*np.mean(alt)),
        "exact_signflip_p_H14_vs_equal_prior":exact_two_sided_signflip(differences),
        "exact_signflip_p_H14_vs_native":exact_two_sided_signflip(rawdiff),
        "exact_signflip_p_H14_vs_H12":exact_two_sided_signflip(alt),
        "bootstrap_95_ci_pp_equal_prior":np.quantile(cluster_boot,[.025,.975]).tolist(),
        "physical_system_rows":rows,
        "scientific_limitations":[
            "New analytical PDEs were authored by investigator, not randomly drawn from nature.",
            "H14 closure prior excludes physically valid damped wave RHS u_t by construction.",
            "A false/absent candidate is counted as error, including abstentions and crashes.",
            "One optimizer seed per PDE and exactly identical per-arm search budget.",
            "H14 and baseline causal closure are postselectors: NO extra EPDE searches.",
            "Even p<0.05 here supports only a controlled synthetic setting, not universal physics.",
            "No tuning of H14 after seeing these 11 results is permitted.",
            "Derivative fields may share finite-difference correlations within training grid.",
        ],
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--records",type=Path,required=True)
    p.add_argument("--source-sha",required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    paths=sorted(args.records.rglob("native-full.json"))
    data=analyze([json.loads(path.read_text(encoding="utf-8")) for path in paths],
                 research_sha=args.source_sha)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(data,ensure_ascii=False,indent=2,
                                      sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({key:data[key] for key in (
        "status","verified_full_searches","baseline_native_exact",
        "baseline_equal_prior_exact","H12_native_exact",
        "H14_closed_exact","H14_abstentions",
        "H14_minus_equal_prior_pp",
        "exact_signflip_p_H14_vs_equal_prior")}))
if __name__=="__main__":
    main()
