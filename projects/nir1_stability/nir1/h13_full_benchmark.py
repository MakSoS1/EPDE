"""H13 parallel no-oracle identifiability audit for exactly 36 physical systems.

H13 is NOT a standalone EPDE equation-discovery algorithm. It receives the
same observed fields as H11/H12 but does NOT rerun evolution, consume labels,
or declare any 'exact recovery' predictions. Report continuous rank evidence.
Never silently omit an unsupported geometry or failed diagnostic.

For scalar 1D PDEs the universal, FIXED exploratory library is
[u, u_x, u_xx, u*u_x]. It is not selected based on the true PDE.
For ODEs the universal, FIXED state monomials are [all state variables,
 their squares, pairwise products], without temporal derivatives.
These feature groups differ in meaning, so never pool their singular
values into a single accuracy metric or claim structural identifiability
of the entire EPDE token search space.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .analyze_full_quality import COHORTS
from .h13_rank_certificate import normalized_spectrum


def one_certificate(name: str) -> dict:
    from projects.pic.epde_bench.datasets import load

    p=load(name)
    arrays={k:np.asarray(v,dtype=float) for k,v in p.data.items()}
    out={"dataset":name,"kind":p.kind,"data_shape":list(p.shape),
         "field_variables":list(arrays),"truth_used":False,
         "only_existing_measurements":True}
    if len(p.grids)==2 and p.dim==1 and len(arrays)==1:
        label,u=next(iter(arrays.items()))
        if u.ndim!=2:
            raise ValueError("1-space PDE expected a 2D time x space field")
        x=np.asarray(p.grids[1],dtype=float)[0,:]
        if not np.all(np.diff(x)>0):
            raise ValueError("Spatial mesh must increase monotonically")
        margin=8
        if u.shape[0]<=2*margin or u.shape[1]<=2*margin+4:
            raise ValueError("Not enough original field for interior crop")
        ux=np.gradient(u,x,axis=1,edge_order=2)
        uxx=np.gradient(ux,x,axis=1,edge_order=2)
        sl=(slice(margin,-margin),slice(margin,-margin))
        X=np.column_stack([v[sl].ravel() for v in
                           (u,ux,uxx,u*ux)])
        out["basis"]=("u","u_x","u_xx","u*u_x")
        out["type"]="universal_scalar_1D_PDE_four_token_spectrum"
        out["certificate"]=normalized_spectrum(X,names=out["basis"])
    elif p.dim==0 and all(np.asarray(v).ndim==1 for v in arrays.values()):
        if len(arrays)>3:
            out["status"]="not_applicable"
            out["reason"]="ODE has more than three coupled variables: fixed state basis undefined"
            return out
        cols=[arrays[key] for key in arrays]
        keys=list(arrays)
        basis=keys[:]
        cols.extend(arrays[k]**2 for k in keys)
        basis.extend(f"{k}^2" for k in keys)
        for i in range(len(keys)):
            for j in range(i+1,len(keys)):
                cols.append(arrays[keys[i]]*arrays[keys[j]])
                basis.append(keys[i]+"*"+keys[j])
        if len(cols)<2:
            out["status"]="not_applicable"
            out["reason"]="Less than two state monomial columns"
            return out
        X=np.column_stack([c.ravel() for c in cols])
        out["basis"]=basis
        out["type"]="universal_ODE_state_monomial_spectrum"
        out["certificate"]=normalized_spectrum(X,names=basis)
    else:
        out["status"]="not_applicable"
        out["reason"]="Only one-dimensional single-field PDE and <=3-variable ODE supported"
        return out
    out["status"]="ok"
    return out


def study():
    records=[]
    for cohort, cases in COHORTS.items():
        for dataset in cases:
            try:
                row=one_certificate(dataset)
            except Exception as exc:
                row={"dataset":dataset,"status":"error","error":repr(exc),
                     "truth_used":False,
                     "only_existing_measurements":True}
            row["cohort"]=cohort
            records.append(row)
    if len(records)!=36 or len({x["dataset"] for x in records})!=36:
        raise ValueError("Missing or duplicated physical benchmark system")
    bycohort={}
    for cohort in COHORTS:
        rows=[r for r in records if r["cohort"]==cohort]
        bycohort[cohort]={
            "systems":len(rows),
            "diagnostic_ok":sum(r["status"]=="ok" for r in rows),
            "not_applicable":sum(r["status"]=="not_applicable" for r in rows),
            "diagnostic_error":sum(r["status"]=="error" for r in rows),
            "rank_ratios":[{"dataset":r["dataset"],
                            "sigma_min_over_max":r["certificate"]["sigma_min_over_max"],
                            "basis_type":r["type"]} for r in rows if r["status"]=="ok"],
        }
    return {
        "status":"H13_PARALLEL_DIAGNOSTIC_NOT_EQUATION_DISCOVERY",
        "expected_physical_systems":36,
        "observed_systems":len(records),
        "input":"same analytical measurement arrays used by full EPDE",
        "extra_observations":0,
        "EPDE_evolution_runs":0,
        "source_of_ground_truth":"not read or used",
        "matched_search_accuracy":None,
        "rank_threshold":None,
        "notes":[
            "H13 cannot compete with H11/H12 on exact-selected accuracy as-is.",
            "Dimensionless singular ratios cannot establish a unique PDE law.",
            "1D PDE basis [u,ux,uxx,uux] and ODE monomials are not comparable.",
            "No tuned threshold, abstention model, calibration or selective accuracy.",
            "Numerical derivatives use existing grid data, interior crop 8.",
            "Missing/unsupported geometry and diagnostic exceptions reported.",
        ],
        "cohorts":bycohort,"records":records,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True,type=Path)
    opt=parser.parse_args()
    data=study()
    opt.output.parent.mkdir(parents=True,exist_ok=True)
    opt.output.write_text(json.dumps(data,indent=2,ensure_ascii=False,
                                      sort_keys=True,allow_nan=False)+"\n",
                          encoding="utf-8")
    print(json.dumps({k:{"ok":v["diagnostic_ok"],
                          "unsupported":v["not_applicable"],
                          "error":v["diagnostic_error"]}
                      for k,v in data["cohorts"].items()}))
    return 0 if all(v["diagnostic_error"]==0
                    for v in data["cohorts"].values()) else 2


if __name__=="__main__":
    raise SystemExit(main())
