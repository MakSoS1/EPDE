"""Candidate-pool discovery across multiple EPDE Pareto fronts.

This *secondary* NIR1 hypothesis uses the SAME five baseline EPDE searches
already generated for the frozen S2 native-selected consensus; it does NOT
launch more evolutions and does NOT require the simulated truth to choose.

A structure may appear on several fronts yet never be selected by any native
PIC compromise. Grouping all fronts by normalized structural identity can
rescue that omission. However this is NOT a free improvement: it needs K
full evolutions and only can recover a law present on one of their fronts.

The rule was specified while S2 was still running (but after its launch).
Results from that S2 remain EXPLORATORY secondary analyses, not a clean S3
confirmatory set for this rule.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from typing import Sequence

from .consensus import _signature, consensus_decision


def pool_consensus_decision(records: Sequence[dict], *, min_repetition: int = 2):
    """Truth-free, deterministic cross-front support selection.

    Ordered priorities:
      1. more independent *restarts* containing the support;
      2. more runs where native PIC selected that support;
      3. lower mean within-run normalized discrepancy regret (zero best);
      4. lower optimizer seed / Pareto index, then lexicographic fingerprint.

    If no support appears in >=min_repetition different fronts, return the
    previously frozen v1 selected-only consensus without looking at truth.
    A tied top structure also follows deterministic priorities, not oracle.
    """
    if min_repetition < 2:
        raise ValueError("min_repetition must be >= 2")
    prior = consensus_decision(records, min_repetition=min_repetition)
    ordered = sorted(records, key=lambda row: int(row["seed"]))
    observed: dict[str, list[dict]] = defaultdict(list)
    for row in ordered:
        seed = int(row["seed"])
        front, objectives = row.get("front"), row.get("objectives")
        if not isinstance(front, list) or not isinstance(objectives, list) or len(front) != len(objectives) or not front:
            raise ValueError("An EPDE front and objectives must be complete")
        selected = int(row["metrics"]["selected_index"])
        scored = []
        for i, (equations, vector) in enumerate(zip(front, objectives)):
            if not isinstance(equations, list) or not equations or vector is None:
                raise ValueError("Incomplete candidate in full Pareto front")
            if not isinstance(vector, list) or len(vector) < 2 or len(vector) % 2:
                raise ValueError("EPDE objective vector must have [discrepancy, instability] per equation")
            if len(vector) != 2*len(equations):
                raise ValueError("Mismatch between equations and objectives")
            values = [float(v) for v in vector]
            if any(not math.isfinite(v) for v in values):
                raise ValueError("Nonfinite Pareto objective, cannot score equally")
            scored.append((i, equations, sum(values[::2]) / len(equations)))
        lo, hi = min(v for _, _, v in scored), max(v for _, _, v in scored)
        # When front loss is effectively flat, avoid amplifying double-roundoff.
        scale = max(hi-lo, 1e-10 * max(1., abs(hi), abs(lo)))
        # Count a candidate at most once per restart even when multiple
        # equations in the same Pareto front canonicalize to one structure.
        within: dict[str, dict] = {}
        for index, equations, discrepancy in scored:
            key = _signature(equations)
            row_entry = {"seed": seed, "index": index, "key": key,
                         "equations": equations,
                         "selected": index == selected,
                         "regret": max(0., (discrepancy - lo) / scale)}
            prev = within.get(key)
            if (prev is None or
                (row_entry["selected"], -row_entry["regret"], -row_entry["index"]) >
                (prev["selected"], -prev["regret"], -prev["index"])):
                within[key] = row_entry
        for key, item in within.items():
            observed[key].append(item)
    support = {k: v for k, v in observed.items() if len(v) >= min_repetition}
    if not support:
        return {**prior, "method": "nir1_allfront_consensus_v2",
                "allfront_fallback_to_v1": True, "front_run_frequency": 0,
                "candidate_supports_seen": len(observed),
                "truth_used_to_select": False}

    def ordering(entry):
        key, occurrences = entry
        frequency = len(occurrences)
        selected_count = sum(bool(v["selected"]) for v in occurrences)
        regret = sum(v["regret"] for v in occurrences) / frequency
        first = min((v["seed"], v["index"]) for v in occurrences)
        return (-frequency, -selected_count, regret, first[0], first[1], key)

    top_key, items = min(support.items(), key=ordering)
    best_item = min(items, key=lambda item: (item["seed"], item["index"]))
    return {
        "method": "nir1_allfront_consensus_v2",
        "n_full_searches": len(ordered),
        "dataset": prior["dataset"], "data_seed": prior["data_seed"],
        "source_revisions": prior["source_revisions"],
        "optimizer_seeds": prior["optimizer_seeds"],
        "chosen_seed": best_item["seed"], "chosen_pareto_index": best_item["index"],
        "chosen_equations": best_item["equations"],
        "chosen_structure_sha256": hashlib.sha256(top_key.encode()).hexdigest(),
        "front_run_frequency": len(items),
        "native_selected_frequency": sum(bool(item["selected"]) for item in items),
        "candidate_supports_seen": len(observed),
        "allfront_fallback_to_v1": False, "all_seeds_accounted": True,
        "truth_used_to_select": False,
        "fidelity": "one output from K identical-budget native full EPDE searches; candidate must be in recorded Pareto front"
    }
