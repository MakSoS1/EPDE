"""Benchmark score recomputation across recursive feature supports (E1)."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time

import numpy as np
from epde.operators.common.sparsity import instability_scores
from epde.operators.common.survival import block_gram_partition


def benchmark(*, n=24000, p=8, repetitions=3, seed=3033, metric="nir1_protected"):
    if n < 300 or p < 3 or repetitions < 2:
        raise ValueError("Invalid E1-RFE benchmark dimensions")
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    X[:, 1] = .9 * X[:, 0] + .43 * X[:, 1]
    w = rng.uniform(.5, 1.5, size=n)
    y = X[:, 0] * .8 + X[:, -1] * .45 + .05*rng.normal(size=n)
    shape = (40, n // 40) if n % 40 == 0 else (n,)
    supports = []
    for size in range(p, 1, -1):
        mask = np.zeros(p + 1, dtype=bool)
        mask[:size] = True
        mask[-1] = size % 2 == 0
        supports.append(mask)
    # Both pathways see the *same* candidate supports and exact data.
    def direct():
        return [instability_scores(metric, X, y, w, shape, mask, p)
                for mask in supports]

    def cached():
        blocks = block_gram_partition(X, y, w, shape, 16, return_yy=True)
        return [instability_scores(metric, X, y, w, shape, mask, p,
                                   nir1_full_blocks=blocks)
                for mask in supports]

    ref, opt = direct(), cached()
    for aa, bb in zip(ref, opt):
        np.testing.assert_allclose(aa, bb, rtol=1e-6, atol=2e-8)
    times = {"direct": [], "precomputed_blocks": []}
    for iteration in range(repetitions):
        methods = (("direct", direct), ("precomputed_blocks", cached))
        if iteration % 2:
            methods = methods[::-1]
        for label, fn in methods:
            t0 = time.perf_counter()
            rows = fn()
            times[label].append(time.perf_counter()-t0)
            for a, b in zip(ref, rows):
                np.testing.assert_allclose(a, b, rtol=1e-6, atol=2e-8)
    med = {name: statistics.median(v) for name, v in times.items()}
    return {"type": "E1_RFE_SCORING_BENCHMARK_NOT_FULL_EPDE",
            "samples": n, "features": p, "supports": len(supports),
            "seed": seed, "metric": metric, "repetitions": repetitions,
            "python": platform.python_version(), "numpy": np.__version__,
            "times_seconds": times, "median_seconds": med,
            "speedup_ratio": med["direct"]/med["precomputed_blocks"],
            "all_supports_parity": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=24000)
    parser.add_argument("--features", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    record = benchmark(n=args.samples, p=args.features,
                       repetitions=args.repetitions)
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"speedup_ratio": record["speedup_ratio"],
                      "parity": record["all_supports_parity"],
                      "median": record["median_seconds"]}))


if __name__ == "__main__":
    main()
