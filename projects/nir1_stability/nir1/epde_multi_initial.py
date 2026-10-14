"""H8 native EPDE experiment with independent initial conditions.

Unlike the fixed-target controlled study, this uses the REAL EPDE evolutionary
search, its original full token library and unrestricted target selection.
It is deliberately only a scientific PILOT: multiple trajectories give extra
information/observations, so a single-trajectory baseline is not same-data
or same-total-sample even if its search budget (16x5) is identical.

No model coefficients or equation truth are passed into search.fit.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from .multitrajectory_identifiability import make_observations


SUPPORTED={
    "nir1_advection": "advection",
    "nir1_heat": "heat",
    "nir1_reactiondiff": "reactiondiff",
    "nir1_wave": "wave",
    "nir1_advdiff": "advdiff",
}


def run_native(case: str, *, arm: str, seed: int,
               smoke: bool = True, initial_seed: int = 11):
    from epde import EpdeSearch
    from projects.pic.epde_bench import env, metrics
    from projects.pic.epde_bench.config import load_config, resolve_for_problem, split_bench_tokens
    from projects.pic.epde_bench.datasets import load
    from projects.pic.epde_bench.runner import pareto_front
    from projects.pic.epde_bench.preprocessing import make_preprocessor
    from projects.pic.epde_bench.tokens import stored_groups
    from projects.pic.epde_bench.paths import REPO_ROOT

    if case not in SUPPORTED or arm not in {"single","multi"}:
        raise ValueError("Unknown dataset or arm")
    env.seed_everything(seed)
    original=load(case)
    cfg=load_config(case,"default")
    sc=resolve_for_problem(cfg,original)
    if smoke:
        sc["evolution"]={"population_size":4,"training_epochs":1}
    elif (sc["evolution"]["population_size"] != 16 or
          sc["evolution"]["training_epochs"] != 5):
        raise ValueError("Scientific full-search budget unexpectedly changed")

    # The benchmark may declare token families requiring externally instantiated
    # tensors/callables; normal PIC always separates them before EpdeSearch.
    sc, benchmark_families = split_bench_tokens(sc)

    # Simulator knows physical law; estimator sees only noisy-free raw
    # trajectory observations on a common grid and matching time axes.
    trajectories=make_observations(SUPPORTED[case],initial_seed)[:(1 if arm=="single" else 3)]
    # Each domain point is shared, but trajectory observations differ.
    # A moderate downsample controls the memory footprint and leaves the
    # dynamics/invariant governing PDE intact.
    t=trajectories[0][0][::3]
    x=trajectories[0][1][::2]
    T,X=__import__("numpy").meshgrid(t,x,indexing="ij")
    search=EpdeSearch(config=sc)
    _,domain=search.createDomain([T,X],ID=0)
    samples=[]
    for i,(_,_,u) in enumerate(trajectories):
        arrays={"u":u[::3,::2].copy()}
        _,tr=search.createTrajectory(arrays,domain,cache_id=i+100,
                                     preprocessor=make_preprocessor(sc))
        samples.append(tr)
    families=benchmark_families + stored_groups(original)
    if original.extra_tokens is not None:
        families+=original.extra_tokens(original)
    record={"dataset":case,"arm":arm,"seed":seed,"initial_seed":initial_seed,
            "search_config":sc,"smoke_only":bool(smoke),
            "n_trajectories":len(samples),"total_observations":sum(
                int(t[2][::3,::2].size) for t in trajectories),
            "time_shape":len(t),"space_shape":len(x),
            "environment":env.environment_info(REPO_ROOT),
            "started":datetime.now(timezone.utc).isoformat(timespec="seconds")}
    started=time.perf_counter()
    try:
        search.fit(data=samples,additional_tokens=families)
        front,objectives=pareto_front(search)
        score=metrics.score_run(front,objectives,original.truth_systems)
        idx=score.get("selected_index")
        record.update(status="ok",front=front,objectives=objectives,
                      metrics=score,selected=(front[idx] if idx is not None else None))
    except Exception as exc:
        record.update(status="error",error=repr(exc),traceback=traceback.format_exc())
    record["elapsed_seconds"]=time.perf_counter()-started
    record["scientific_fairness_note"]=(
        "Same EPDE evolution budget and token library, but multi arm has "
        "3x more trajectories and more total samples; NOT equal information.")
    return record


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset",required=True,choices=sorted(SUPPORTED))
    parser.add_argument("--arm",required=True,choices=("single","multi"))
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--initial-seed",type=int,default=11)
    parser.add_argument("--full",action="store_true")
    parser.add_argument("--output",required=True,type=Path)
    opt=parser.parse_args(argv)
    result=run_native(opt.dataset,arm=opt.arm,seed=opt.seed,
                      smoke=not opt.full,initial_seed=opt.initial_seed)
    opt.output.parent.mkdir(parents=True,exist_ok=True)
    opt.output.write_text(json.dumps(result,sort_keys=True,indent=2,
                                      allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"dataset":opt.dataset,"arm":opt.arm,
                      "status":result["status"],
                      "exact":(result.get("metrics") or {}).get("success_selected"),
                      "front_exact":(result.get("metrics") or {}).get("success_front"),
                      "elapsed_s":result["elapsed_seconds"]}))
    return 0 if result["status"]=="ok" else 2

if __name__=="__main__":
    raise SystemExit(main())
