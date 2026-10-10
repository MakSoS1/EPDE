"""Strict NIR1 E1 cached-vs-reference complete EPDE parity gate.

This checks identical PIC budgets and seed, structural Pareto support and
selected equation. No new scientific accuracy is inferred from this test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from projects.pic.epde_bench.metrics import canonical_tokens


def compare(cached: dict, reference: dict):
    if cached["status"] != "ok" or reference["status"] != "ok":
        raise AssertionError("Both full EPDE searches must succeed")
    for field in ("dataset", "seed", "config", "search_config"):
        if cached.get(field) != reference.get(field):
            raise AssertionError(f"Unmatched full EPDE experimental configuration: {field}")
    if cached.get("research_variant") != reference.get("research_variant"):
        raise AssertionError("Unmatched research variant")
    def canonical_front(record):
        return sorted([repr(canonical_tokens(eq)) for eq in record["front"]])
    fa, fb = canonical_front(cached), canonical_front(reference)
    if fa != fb:
        raise AssertionError("Pareto-front structure differs; speed optimization not approved")
    if (canonical_tokens(cached.get("selected") or []) !=
            canonical_tokens(reference.get("selected") or [])):
        raise AssertionError("PIC compromise selection differs after acceleration")
    ma, mb = cached["metrics"], reference["metrics"]
    for field in ("success_selected", "success_front", "hamming_selected", "hamming_min"):
        if ma.get(field) != mb.get(field):
            raise AssertionError(f"Structural metric changed: {field}")
    # Front score rows may be reordered with EPDE's tie resolution.
    # Compare each value vector paired with its full printed equation set.
    def paired_scores(rec):
        return {repr(canonical_tokens(eq)): val for eq, val in
                zip(rec["front"], rec["objectives"])}
    A, B = paired_scores(cached), paired_scores(reference)
    if A.keys() != B.keys():
        raise AssertionError("Objective vectors cannot be matched to structures")
    maximum_error = 0.
    for key in A:
        va, vb = np.asarray(A[key], dtype=float), np.asarray(B[key], dtype=float)
        if not np.allclose(va, vb, rtol=1e-5, atol=1e-7):
            raise AssertionError(f"Objective changed after exact Gram reuse: {key}")
        maximum_error = max(maximum_error, float(np.max(np.abs(va-vb))))
    return {
        "status": "PARITY_PASSED", "samples": len(A),
        "same_candidate_support": True, "same_selected_structure": True,
        "same_structural_success": True, "max_abs_objective_difference": maximum_error,
        "full_budget": cached["search_config"]["evolution"],
        "cached_wall_s": cached["total_seconds"],
        "reference_wall_s": reference["total_seconds"],
        "speed_ratio_one_pair_not_stable_benchmark": (
            reference["total_seconds"] / cached["total_seconds"]),
        "experiment_scope": "SAME_SYSTEM_SEED_ONLY_NOT_GLOBAL_QUALITY_GUARANTEE",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cached", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cache = json.loads(args.cached.read_text(encoding="utf-8"))
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    result = compare(cache, reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
