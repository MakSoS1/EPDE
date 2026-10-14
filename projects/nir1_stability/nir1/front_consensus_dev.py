"""S1 and S1b development-only replay of all-front consensus (no new evolution)."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .campaign import resume_plan
from .consensus import consensus_decision, min_discrepancy_restart
from .front_consensus import pool_consensus_decision
from .metrics import _cluster_signflip_p
from .records import read_record


def analyze_dev(manifest, paths):
    if manifest["stage"] != "S1":
        raise ValueError("S1b development replay expected")
    permitted = [
        {"default", "nir1_sparsefront"},
        {"default", "nir1_criterion_only", "nir1_regulator_only", "nir1_combined"},
    ]
    if {s["method"] for s in manifest["runs"]} not in permitted:
        raise ValueError("Only original frozen S1 and S1b factorial arms are valid")
    ledger = resume_plan(manifest, paths)
    if (ledger["remaining"] or ledger["invalid_records"] or
            ledger["status_counts"]["ok"] != ledger["planned"]):
        return {"status": "INCOMPLETE_OR_ERROR", "ledger": ledger}
    from projects.pic.epde_bench.datasets import load
    from projects.pic.epde_bench.metrics import canonical_tokens, hamming_best

    latest = {}
    for p in paths:
        row = read_record(p)
        if row["run_id"] in latest:
            raise ValueError("Unresolved duplicate records")
        latest[row["run_id"]] = row
    groups = defaultdict(list)
    for run in manifest["runs"]:
        if run["method"] != "default":
            continue
        rec = latest[run["run_id"]]
        child = rec["rows"][0]
        if child.get("research_variant") != "default" or child["seed"] != run["optimizer_seed"]:
            raise ValueError("S1b baseline identity mismatch")
        groups[(run["case"], run["data_seed"])].append(child)
    rows = []
    for (case, dataseed), records in sorted(groups.items()):
        # Both algorithms must select with NO access to hidden simulator truth.
        c1 = consensus_decision(records)
        c2 = pool_consensus_decision(records)
        equal = min_discrepancy_restart(records)
        accepted = [canonical_tokens(s) for s in load(
            case, **records[0].get("config", {}).get("loader", {})).truth_systems]
        v1 = hamming_best(canonical_tokens(c1["chosen_equations"]), accepted) == 0
        v2 = hamming_best(canonical_tokens(c2["chosen_equations"]), accepted) == 0
        matched_cost = hamming_best(canonical_tokens(equal["chosen_equations"]), accepted) == 0
        native = sum(int(r["metrics"]["success_selected"] is True)
                     for r in records)
        rows.append({"system": case, "data_seed": dataseed,
                     "n_native_full_searches": len(records),
                     "native_exact_count": native,
                     "v1_selected_only_consensus_exact": bool(v1),
                     "v2_allfront_consensus_exact": bool(v2),
                     "equal_cost_native_min_discrepancy_exact": bool(matched_cost),
                     "equal_cost_chosen_seed": int(equal["chosen_seed"]),
                     "v2_fallback": c2["allfront_fallback_to_v1"],
                     "v2_support_run_frequency": c2["front_run_frequency"]})
    diff = np.array([float(r["v2_allfront_consensus_exact"]) -
                     float(r["v1_selected_only_consensus_exact"])
                     for r in rows])
    diff_equal = np.array([float(r["v1_selected_only_consensus_exact"]) -
                           float(r["equal_cost_native_min_discrepancy_exact"])
                           for r in rows])
    p, _ = _cluster_signflip_p(diff, seed=20261011)
    p_equal, _ = _cluster_signflip_p(diff_equal, seed=20261012)
    return {"status": "COMPLETE_DEVELOPMENT_ONLY_NOT_INDEPENDENT_VALIDATION",
            "n_independent_systems": len(rows),
            "planned_full_searches": ledger["planned"],
            "native_single_success": float(sum(r["native_exact_count"] for r in rows)/
                                           sum(r["n_native_full_searches"] for r in rows)),
            "selected_only_consensus_success": float(np.mean([
                r["v1_selected_only_consensus_exact"] for r in rows])),
            "allfront_consensus_success": float(np.mean([
                r["v2_allfront_consensus_exact"] for r in rows])),
            "equal_cost_native_min_discrepancy_success": float(np.mean([
                r["equal_cost_native_min_discrepancy_exact"] for r in rows])),
            "v1_minus_equal_cost_delta_pp": 100*float(np.mean(diff_equal)),
            "v1_minus_equal_cost_cluster_p": float(p_equal),
            "allfront_minus_selected_consensus_pp": 100*float(np.mean(diff)),
            "cluster_exact_signflip_p": float(p),
            "manifest_sha": manifest["manifest_sha"], "source_code_sha": manifest["code_sha"],
            "science_warning": "S1 and S1b archives were already inspected before formulation; "
                               "development-only, must not tune on S2 and claim confirmation",
            "rows": rows, "ledger": ledger}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    paths = sorted(p for p in args.records.rglob("*.json")
                   if len(p.stem) == 64 and all(ch in "0123456789abcdef" for ch in p.stem))
    result = analyze_dev(manifest, paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True,
                                      allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"status": result["status"],
                      "delta_pp": result.get("allfront_minus_selected_consensus_pp")}))
    return 0 if result["status"].startswith("COMPLETE") else 2


if __name__ == "__main__":
    raise SystemExit(main())
