"""H8 identifiability certificate using the *same* data/observation budgets.

Report only dataset geometry, Gram eigenvalues and unique explained energy.
No benchmark truth is needed to compute the certificate. The fixed pair
(u, u_xx) is applied to EVERY scalar PDE, not chosen by looking at failures.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from .multitrajectory_identifiability import make_observations, fixed_dictionary

PDE_CASES=("advection","heat","reactiondiff","wave","advdiff")


def gram_diagnostics(X: np.ndarray, names):
    cols=[names.index("u"),names.index("uxx")]
    A=np.asarray(X,dtype=np.float64)[:,cols]
    norms=np.linalg.norm(A,axis=0)
    if np.any(norms <= 0):raise ValueError("Zero-energy feature invalidates Gram")
    B=A/norms
    gram=B.T@B
    eig=np.linalg.eigvalsh(gram)
    corr=float(gram[0,1])
    # For two features unique energy Q=1-R², the same for both.
    quality=float(np.clip(1.-corr*corr,0.,1.))
    return {"feature_pair":["u","uxx"],"unique_energy":quality,
            "minimum_normalized_gram_eigenvalue":float(eig[0]),
            "maximum_normalized_gram_eigenvalue":float(eig[-1]),
            "condition_number":(float(eig[-1]/eig[0])
                                if eig[0]>1e-14 else None),
            "samples":len(X),
            "unstable_or_unidentifiable":bool(eig[0]<1e-8)}


def study(seed: int = 0):
    rows=[]
    for name in PDE_CASES:
        order=2 if name=="wave" else 1
        data=[fixed_dictionary(row,target_time_order=order)
              for row in make_observations(name,seed)[:3]]
        single=data[0][1][::3]
        multi=np.concatenate([row[1][::9] for row in data],axis=0)
        if abs(len(single)-len(multi))>3:
            raise AssertionError("Total observation budget mismatch")
        rows.append({"case":name,"seed":seed,
                     "single":gram_diagnostics(single,data[0][0]),
                     "multi":gram_diagnostics(multi,data[0][0]),
                     "same_training_observations_within_three":True})
    return {"type":"IDENTIFIABILITY_DIAGNOSTIC_NOT_EPDE_ACCURACY",
            "scientific_prior":"known derivative order, same pair (u,uxx) on every PDE",
            "rows":rows}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--seed",type=int,default=0)
    args=p.parse_args()
    report=study(args.seed)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,sort_keys=True,
                                      allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps([{ "case":r["case"],
                        "single_q":r["single"]["unique_energy"],
                        "multi_q":r["multi"]["unique_energy"]}
                     for r in report["rows"]]))


if __name__=="__main__":
    main()
