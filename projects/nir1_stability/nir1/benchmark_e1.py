"""Empirical, reproducible E1 speed test; no result is claimed in advance.

Run under OPENBLAS_NUM_THREADS=1 / OMP_NUM_THREADS=1 to reduce VM noise.
Print stdout JSON, with exact parity and runtime samples preserved.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time

import numpy as np
from epde.operators.common.survival import heterogeneity_scores, nir1_excess_scores


def old_score(X, y, w, grid_shape):
    """Original NIR1 v1, before Gram reuse (independent full-data matmul)."""
    n, p = X.shape
    excess = heterogeneity_scores(X, y, w, grid_shape, fit_intercept=True)
    A = np.column_stack([X, np.ones(n)])
    G = A.T @ (w[:, None] * A)
    norm = np.sqrt(np.maximum(np.diag(G), 0.))
    scale = np.where(norm > 0., norm, 1.)
    normalized = G / np.outer(scale, scale)
    quality = np.zeros(p)
    for j in range(p):
        if norm[j] <= 0:
            continue
        other = np.arange(normalized.shape[0]) != j
        projection = np.linalg.lstsq(normalized[np.ix_(other, other)],
                                     normalized[other, j], rcond=1e-10)[0]
        quality[j] = np.clip(1 - normalized[other, j] @ projection / normalized[j, j],
                             0., 1.)
    return np.where(norm[:p] > 0, np.clip((excess + (1-quality))/2, 0, 1), 1.)


def benchmark(*, samples: int = 12000, features: int = 7,
              repetitions: int = 5, seed: int = 4219):
    if samples < 400 or features < 2 or not 3 <= repetitions <= 30:
        raise ValueError("Need >=400 samples, >=2 features, 3-30 measurements")
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(samples, features))
    X[:, 1] = .88 * X[:, 0] + .48 * X[:, 1]
    w = rng.uniform(.7, 1.3, size=samples)
    y = 1.2 * X[:, 0] - .8 * X[:, -1] + .2 + .1 * rng.normal(size=samples)
    shape = (40, samples // 40) if samples % 40 == 0 else (samples,)
    funcs = {"legacy": lambda: old_score(X, y, w, shape),
             "reuse_block_gram": lambda: nir1_excess_scores(X, y, w, shape)}
    for func in funcs.values():
        func()
    results = {}
    outputs = {}
    # Alternate order per iteration: controls for monotonic VM/BLAS warmup.
    for iteration in range(repetitions):
        for name in (("legacy", "reuse_block_gram") if iteration % 2 == 0
                     else ("reuse_block_gram", "legacy")):
            start = time.perf_counter()
            outputs[name] = funcs[name]()
            results.setdefault(name, []).append(time.perf_counter() - start)
    a, b = outputs["legacy"], outputs["reuse_block_gram"]
    max_abs = float(np.max(np.abs(a - b)))
    assert np.allclose(a, b, rtol=2e-7, atol=1e-8), (
        f"Exact-block-Gram parity failed; maximum difference {max_abs}")
    reference_time = statistics.median(results["legacy"])
    optimized_time = statistics.median(results["reuse_block_gram"])
    return {"type": "E1_MICROBENCHMARK_NOT_FULL_EPDE", "samples": samples,
            "features": features, "seed": seed, "repetitions": repetitions,
            "timings_s": results, "medians_s": {
                k: statistics.median(v) for k, v in results.items()},
            "speedup_ratio": reference_time / optimized_time,
            "maximum_absolute_score_difference": max_abs,
            "parity_passed": True, "python": platform.python_version(),
            "numpy": np.__version__, "platform": platform.platform()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=12000)
    parser.add_argument("--features", type=int, default=7)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = benchmark(samples=args.samples, features=args.features,
                        repetitions=args.repetitions)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    print(json.dumps({"parity": summary["parity_passed"],
                      "max_delta": summary["maximum_absolute_score_difference"],
                      "speedup": summary["speedup_ratio"],
                      "median_s": summary["medians_s"]}))


if __name__ == "__main__":
    main()
