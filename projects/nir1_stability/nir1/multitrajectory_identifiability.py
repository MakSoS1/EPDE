"""H8: structural identifiability from diverse initial conditions.

This is a CONTROLLED fixed-target sparse-regression experiment, not a claim
about improving unrestricted EPDE evolutionary search. The initial-value
simulator contains physical ground truth; the sparse selector receives ONLY
observation/derivative arrays, the predeclared dictionary and a known target
derivative order. No truth terms or coefficients enter the estimator.

Single and multi arms have the same number of training/validation samples.
The fourth trajectory is held out until AFTER both sparse supports are
chosen. All four trajectories share the same governing equation.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

CASES = (
    "decay", "logistic", "quadratic", "cubic", "harmonic",
    "advection", "heat", "reactiondiff", "wave", "advdiff",
)
# Used ONLY in offline evaluation after both selections; never a selector arg.
TRUTH = {
    "decay": ("u",),
    "logistic": ("u", "u2"),
    "quadratic": ("u2",),
    "cubic": ("u", "u3"),
    "harmonic": ("u",),
    "advection": ("ux",),
    "heat": ("uxx",),
    "reactiondiff": ("u", "uxx"),
    "wave": ("uxx",),
    "advdiff": ("ux", "uxx"),
}
SECOND_ORDER_TARGETS = frozenset(("harmonic", "wave"))
PENALTY = 0.0005
MAX_SUPPORT_SIZE = 3


def make_observations(case: str, seed: int):
    """Four analytically generated trajectories, different initial conditions.

    The test-case name is used ONLY in the simulator. The estimator does not
    receive it: it receives feature matrices, derivatives and known time order.
    """
    if case not in CASES:
        raise ValueError("Unsupported analytic study case")
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 5, 241, dtype=np.float64)
    x = np.linspace(-np.pi, np.pi, 112, endpoint=False, dtype=np.float64)
    result = []
    for j in range(4):
        amp = 1.0 + rng.uniform(-0.05, 0.05)
        shift = rng.uniform(-0.12, 0.12)
        if case == "decay":
            u = amp * (0.5 + 0.7*j)*np.exp(-0.7*t)
        elif case == "logistic":
            A = [0.005, 1.0, 6.0, 2.5][j]
            u = 2.0/(1.0 + A*np.exp(-0.65*t))
        elif case == "quadratic":
            start = amp * [0.05, 0.6, 1.8, 1.25][j]
            u = 1.0/(1.0/start + 0.22*t)
        elif case == "cubic":
            B = [0.005, 1.0, 7.0, 2.0][j]
            u = 1.0/np.sqrt(1.0 + B*np.exp(-2.0*t))
        elif case == "harmonic":
            u = amp * ((0.2 + 0.5*j)*np.cos(1.3*t) +
                       (0.01 + 0.2*j)*np.sin(1.3*t))
        else:
            tt, xx = np.meshgrid(t, x, indexing="ij")
            mode = j+1
            if case == "advection":
                u = amp*np.sin(mode*(xx-0.7*tt) + shift)
            elif case == "heat":
                u = amp*np.sin(mode*xx+shift)*np.exp(-0.17*mode**2*tt)
            elif case == "reactiondiff":
                u = amp*np.sin(mode*xx+shift)*np.exp((0.35-0.19*mode**2)*tt)
            elif case == "wave":
                u = amp*(np.sin(mode*xx+shift)*np.cos(0.8*mode*tt) +
                         0.24*np.cos((mode+1)*xx+shift) *
                         np.sin(0.8*(mode+1)*tt))
            elif case == "advdiff":
                u = amp*np.sin(mode*(xx-0.65*tt)+shift)*np.exp(-0.13*mode**2*tt)
        result.append((t,x,u))
    return result


def fixed_dictionary(observation, *, target_time_order: int):
    """Observed (feature matrix, target derivative), never truth labels.

    A known 1st/2nd time-order is explicitly ASSUMED input information;
    unrestricted EPDE is not given this prior and is NOT a matched baseline.
    """
    t,x,u = observation
    dt=t[1]-t[0]
    ut=np.gradient(u,dt,axis=0,edge_order=2)
    if target_time_order not in (1,2):
        raise ValueError("Known time derivative order must be 1 or 2")
    y=(ut if target_time_order == 1 else
       np.gradient(ut,dt,axis=0,edge_order=2))
    if u.ndim == 1:
        names=("u","u2","u3")
        X=np.stack((u,u*u,u**3),axis=-1)
        sl=np.s_[6:-6]
    elif u.ndim == 2:
        dx=x[1]-x[0]
        ux=np.gradient(u,dx,axis=1,edge_order=2)
        uxx=np.gradient(ux,dx,axis=1,edge_order=2)
        names=("u","u2","u3","ux","uxx","u_ux")
        X=np.stack((u,u*u,u**3,ux,uxx,u*ux),axis=-1)
        sl=np.s_[6:-6,6:-6]
    else:
        raise ValueError("Only scalar ODE and 1D-space PDE observations supported")
    return names, np.asarray(X[sl]).reshape(-1,len(names)), np.asarray(y[sl]).reshape(-1)


def choose_support(training: list, validation: list, *,
                   penalty: float = PENALTY, max_support: int = MAX_SUPPORT_SIZE):
    """Truth-free sparse LS with fixed validation+complexity criterion.

    Fitting and all support decisions use only observation derivatives and
    specified dictionary; the selected model is evaluated on a FOURTH
    trajectory later and no coefficient is refit there.
    """
    if not training or not validation or penalty < 0 or max_support < 1:
        raise ValueError("Invalid sparse model selection configuration")
    names=training[0][0]
    if any(part[0] != names for part in training+validation):
        raise ValueError("Dictionary mismatch across initial conditions")
    Xt=np.concatenate([part[1] for part in training],axis=0)
    yt=np.concatenate([part[2] for part in training],axis=0)
    Xv=np.concatenate([part[1] for part in validation],axis=0)
    yv=np.concatenate([part[2] for part in validation],axis=0)
    if Xt.shape[1] != len(names) or Xt.shape[0] < 10 * len(names):
        raise ValueError("Insufficient observations")
    best=None
    for cardinality in range(1,min(max_support,len(names))+1):
        for support in itertools.combinations(range(len(names)),cardinality):
            design=Xt[:,support]
            norm=np.linalg.norm(design,axis=0)
            if np.any(norm < 1e-14):
                continue
            coeff=np.linalg.lstsq(design/norm,yt,rcond=None)[0]/norm
            mse=float(np.mean((Xv[:,support]@coeff-yv)**2))
            relative=mse/(float(np.var(yv))+1e-12)
            score=relative+penalty*cardinality
            row=(score,support,relative,coeff)
            if best is None or row[0]<best[0]-1e-14:
                best=row
    if best is None:
        raise ValueError("All candidate libraries numerically degenerate")
    _,support,relative,beta=best
    # Local identifiability of active columns in TRAINING data, not oracle.
    A=Xt[:,support]
    norms=np.linalg.norm(A,axis=0)
    gram=(A/norms).T@(A/norms)
    eigval=np.linalg.eigvalsh(gram)
    return {"support":tuple(names[i] for i in support),
            "indices":tuple(support),
            "coefficients":beta,
            "validation_relative_mse":relative,
            "score":best[0],
            "min_normalized_gram_eigenvalue":float(eigval[0]),
            "max_normalized_gram_eigenvalue":float(eigval[-1]),
            "train_samples":len(yt),
            "validation_samples":len(yv)}


def one_study(case: str, seed: int):
    observations=make_observations(case,seed)
    order=2 if case in SECOND_ORDER_TARGETS else 1
    parts=[fixed_dictionary(row,target_time_order=order) for row in observations]
    # Explicit equal-observation-budget split (3× samples from one
    # trajectory vs 3× trajectories at 1/3 sampling rate each).
    one_train=[(parts[0][0],parts[0][1][::3],parts[0][2][::3])]
    one_val=[(parts[0][0],parts[0][1][1::3],parts[0][2][1::3])]
    multi_train=[(q[0],q[1][::9],q[2][::9]) for q in parts[:3]]
    multi_val=[(q[0],q[1][1::9],q[2][1::9]) for q in parts[:3]]
    a=choose_support(one_train,one_val)
    b=choose_support(multi_train,multi_val)
    # Only AFTER both choices are fixed: evaluate independent fourth start.
    def heldout(model):
        q=parts[3];idx=model["indices"]
        yhat=q[1][:,idx]@model["coefficients"]
        return float(np.mean((yhat-q[2])**2)/(np.var(q[2])+1e-12))
    truth=set(TRUTH[case])
    return {"case":case,"seed":seed,"time_order_known":order,
            "single":{"terms":a["support"],"exact":set(a["support"])==truth,
                      "heldout_relative_mse":heldout(a),
                      "min_gram_eig":a["min_normalized_gram_eigenvalue"],
                      "training_observations":a["train_samples"]},
            "multi":{"terms":b["support"],"exact":set(b["support"])==truth,
                     "heldout_relative_mse":heldout(b),
                     "min_gram_eig":b["min_normalized_gram_eigenvalue"],
                     "training_observations":b["train_samples"]},
            "truth_used_for_scoring_only":True,
            "information_note":"Known time derivative order and restricted dictionary are priors given identically to both arms."}


def experiment(seeds=range(5)):
    rows=[one_study(case,seed) for case in CASES for seed in seeds]
    counts={arm:sum(bool(row[arm]["exact"]) for row in rows)
            for arm in ("single","multi")}
    summary={"status":"FIXED_TARGET_IDENTIFIABILITY_NOT_FULL_EPDE",
             "seeds":list(seeds),"cases":list(CASES),
             "n_systems":len(CASES),"n_optimizer_like_repeats":len(rows),
             "correct":counts,"relative_improvement_pp":100*(counts["multi"]-counts["single"])/len(rows),
             "full_rows":rows,
             "scientific_limitations":[
                 "Single-mode initial conditions intentionally confound u and u_xx.",
                 "Known temporal target order and restricted library given to BOTH arms; not equivalent to unrestricted EPDE.",
                 "Three initial conditions deliver different information than a single trajectory despite equal sample count.",
                 "Seeds and related synthetic dynamics are NOT independent physical systems.",
                 "A fourth initial condition is held out for derivative prediction after both selections.",
                 "No superiority claim until independent controlled transfer and EPDE integration."]}
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--seeds",type=int,default=5)
    args=p.parse_args()
    if args.seeds<1 or args.seeds>20:raise ValueError("seed count out of bounds")
    result=experiment(range(args.seeds))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,sort_keys=True,indent=2,
                                      allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"correct":result["correct"],"n":len(result["full_rows"]),
                      "effect_pp":result["relative_improvement_pp"]}))


if __name__=="__main__":
    main()
