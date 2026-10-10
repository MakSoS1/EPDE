"""Measure E1 axis-prefix chi2 support scoring on the same data and masks.

Not an EPDE quality or wall-time claim. Full EPDE structural parity is
tested separately in nir1-full-chi2-parity.yml.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np

from epde.operators.common.chi2_prefix import prepare_chi2_prefix
from epde.operators.common.sparsity import instability_scores


def measure(shape=(160, 160), p=10, repetitions=5, seed=20261011):
    if min(shape) < 3 or p < 4 or repetitions < 3:
        raise ValueError("Invalid profile parameters")
    n=int(np.prod(shape))
    rng=np.random.default_rng(seed)
    X=rng.normal(size=(n,p))
    X[:,1]=0.9*X[:,0]+.43588989*X[:,1]
    w=rng.uniform(.7,1.3,size=n)
    y=1.3*X[:,0]-1.5*X[:,3]+.1*rng.normal(size=n)
    A=np.column_stack((X,np.ones(n)))
    G=A.T@(w[:,None]*A)
    Gy=A.T@(w*y)
    masks=[]
    for count in range(p,1,-1):
        active=np.zeros(p+1,dtype=bool)
        active[:count]=True
        active[-1]=count%2 == 0
        masks.append(active)
    def original():
        return [instability_scores("chi2",X,y,w,shape,mask,p) for mask in masks]
    def accelerated():
        c=prepare_chi2_prefix(X,y,w,shape,G,Gy)
        if c is None:
            raise AssertionError("Benchmark dimensions unexpectedly rejected by prefix cache")
        return [instability_scores("chi2",X,y,w,shape,mask,p,chi2_prefix=c)
                for mask in masks]
    expected=original()
    observed=accelerated()
    for a,b in zip(expected,observed):
        np.testing.assert_allclose(a,b,atol=5e-6,rtol=1e-6)
    timings={"reference":[],"axis_prefix":[]}
    for rep in range(repetitions):
        order=(("reference",original),("axis_prefix",accelerated))
        if rep%2:order=order[::-1]
        for label,fn in order:
            start=time.perf_counter()
            values=fn()
            timings[label].append(time.perf_counter()-start)
            for old,new in zip(expected,values):
                np.testing.assert_allclose(old,new,atol=5e-6,rtol=1e-6)
    median={label:statistics.median(v) for label,v in timings.items()}
    return {
        "kind":"E1_SCORE_CACHED_CHI2_MICROBENCHMARK_NOT_FULL_EPDE",
        "shape":list(shape), "n_samples":n,"features":p,
        "supports":len(masks),"seed":seed,
        "repetitions":repetitions,"reference_parity":True,
        "max_absolute_score_difference":float(max(
            np.max(np.abs(a-b)) for a,b in zip(expected,observed))),
        "timings_seconds":timings,"median_seconds":median,
        "speedup_ratio":median["reference"]/median["axis_prefix"],
        "python":platform.python_version(),"numpy":np.__version__,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--time-levels",type=int,default=160)
    parser.add_argument("--space-levels",type=int,default=160)
    parser.add_argument("--features",type=int,default=10)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=measure(shape=(args.time_levels,args.space_levels),p=args.features)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({"shape":result["shape"],"speedup":result["speedup_ratio"],
                      "parity":result["reference_parity"]}))


if __name__=="__main__":
    main()
