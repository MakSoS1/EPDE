"""Repaired H9/H10 utopia follow-up; DOES NOT change the frozen 756 trials.

These two additional arms answer: does H12 repair remove crashes while keeping
the original normalized-utopia Pareto selector? They are new post-hoc runs,
not replacements for failed H9/H10/H11 records.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback

ARMS = ("anchor_utopia_repair", "mdl_utopia_repair")
FLAGS = ("NIR1_DYNAMIC_TARGET", "NIR1_DYNAMIC_MDL", "NIR1_DYNAMIC_REPAIR")


def execute(dataset: str, arm: str, seed: int, *, data_seed: int = 0,
            smoke: bool = False):
    if arm not in ARMS:
        raise ValueError(f"Unknown repaired control {arm!r}")
    from projects.pic.epde_bench.config import deep_merge
    from projects.pic.epde_bench.runner import run_one
    opts = {"search": {"objectives": {"instability_metric": "chi2",
                                      "sparsity_cls": "vwsr"}},
            "nir1_data_seed": data_seed}
    if smoke:
        opts = deep_merge(opts, {"search": {"evolution": {
            "population_size": 4, "training_epochs": 1}}})
    old = {k: os.environ.get(k) for k in FLAGS}
    try:
        os.environ["NIR1_DYNAMIC_TARGET"] = "1"
        os.environ["NIR1_DYNAMIC_REPAIR"] = "1"
        if arm == "mdl_utopia_repair":
            os.environ["NIR1_DYNAMIC_MDL"] = "1"
        else:
            os.environ.pop("NIR1_DYNAMIC_MDL", None)
        # PIC run_one already selects by the native normalized utopia rule.
        # Ground truth is accessed only to SCORE that chosen index.
        record = run_one(dataset, "default", 0.0, seed, overrides=opts)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    record["factorial_arm"] = arm
    record["factorial_protocol"] = "mdlablation20261014-repair-followup-v1"
    record["research_variant"] = arm
    record["data_seed_requested"] = data_seed
    record["optimizer_seed"] = seed
    record["nir1_smoke_only"] = smoke
    record["selection_strategy"] = "native_normalized_utopia_with_h12_repair"
    record["research_physical_prior"] = (
        "H9 temporal target and H12 repair, native utopia front selector; " +
        ("H10 MDL enabled." if arm == "mdl_utopia_repair"
         else "H10 MDL disabled."))
    return record


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",required=True)
    p.add_argument("--arm",choices=ARMS,required=True)
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--data-seed",type=int,default=0)
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    try:
        r=execute(a.dataset,a.arm,a.seed,data_seed=a.data_seed,smoke=a.smoke)
    except Exception as exc:
        r=dict(status="crash",error=repr(exc),traceback=traceback.format_exc(),
               dataset=a.dataset,factorial_arm=a.arm,optimizer_seed=a.seed,
               data_seed_requested=a.data_seed,nir1_smoke_only=a.smoke,
               factorial_protocol="mdlablation20261014-repair-followup-v1")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    fd,staged=tempfile.mkstemp(prefix=".repair-",dir=a.output.parent)
    try:
        with os.fdopen(fd,"w",encoding="utf8") as f:
            json.dump(r,f,ensure_ascii=False,sort_keys=True,indent=2,
                      default=str,allow_nan=False)
            f.write("\n")
        os.replace(staged,a.output)
    finally:
        if os.path.exists(staged): os.unlink(staged)
    print(json.dumps({"status":r["status"],"arm":a.arm,
                      "dataset":a.dataset,"seed":a.seed}),flush=True)
    return 0 if r["status"]=="ok" else 1


if __name__=="__main__":
    sys.exit(main())
