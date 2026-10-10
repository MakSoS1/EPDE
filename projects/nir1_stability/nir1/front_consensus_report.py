"""Read-only secondary analysis of all-front multi-restart structural consensus.

Uses exactly the immutable, complete baseline-restart manifest already frozen
for NIR1 S2. The new selection rule must not be tuned after S2 disclosure.
Because it was defined WHILE the S2 execution was underway, this remains an
EXPLORATORY secondary analysis, not preregistered primary confirmation.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .consensus import consensus_decision
from .consensus_report import analyze_consensus
from .front_consensus import pool_consensus_decision
from .metrics import _cluster_signflip_p
from .records import read_record


def analyze_front_consensus(manifest: dict, paths: list[Path]) -> dict:
    """Produce one scored decision per physical system, fail closed on loss."""
    # First run the original immutable S2 audit. This is also the gate for
    # missing identities, conflicting retries, checksum failures and cross-
    # system inconsistencies; no retries / timeouts are silently omitted.
    audited = analyze_consensus(manifest, paths)
    if not audited["status"].startswith("COMPLETE"):
        return {"status": "INCOMPLETE_BLOCKS_INFERENCE",
                "manifest_sha": manifest["manifest_sha"],
                "ledger": audited["ledger"]}

    from projects.pic.epde_bench.datasets import load
    from projects.pic.epde_bench.metrics import canonical_tokens, hamming_best

    by_id = {r["run_id"]: r for r in (read_record(path) for path in paths)}
    by_system = defaultdict(list)
    for spec in manifest["runs"]:
        row = by_id[spec["run_id"]]["rows"][0]
        by_system[(spec["case"], int(spec["data_seed"]))].append(row)
    entries = []
    native_by_system = {(r["system"], int(r["data_seed"])): r
                        for r in audited["systems"]}
    for key, records in sorted(by_system.items()):
        case, data_seed = key
        # Both selectors are called and FROZEN before any simulation truth
        # from the dataset loader is accessed.
        old = consensus_decision(records)
        new = pool_consensus_decision(records)
        if old["n_full_searches"] != new["n_full_searches"]:
            raise AssertionError("Unequal search budgets")
        problem = load(case, **records[0].get("config", {}).get("loader", {}))
        if not problem.truth_systems:
            raise ValueError("No known equation truth; cannot score this case")
        truth = [canonical_tokens(system) for system in problem.truth_systems]
        # Must use actual candidate chosen from front, not the source
        # run's default metrics.selected_index (which may differ).
        label_old = bool(hamming_best(canonical_tokens(old["chosen_equations"]), truth) == 0)
        label_new = bool(hamming_best(canonical_tokens(new["chosen_equations"]), truth) == 0)
        native = native_by_system[key]
        if label_old != native["consensus_selected_exact"]:
            raise AssertionError("Scoring rule disagrees with original S2 consensus audit")
        entries.append({
            "system": case, "data_seed": data_seed,
            "native_pic_success_fraction": native["baseline_rate"],
            "v1_selected_only_consensus_exact": label_old,
            "v2_allfront_consensus_exact": label_new,
            "v1_chosen_seed": old["chosen_seed"],
            "v2_chosen_seed": new["chosen_seed"],
            "v2_chosen_pareto_index": new.get("chosen_pareto_index"),
            "v2_candidate_supports_seen": new["candidate_supports_seen"],
            "v2_support_run_frequency": new["front_run_frequency"],
            "v2_selected_run_frequency": new.get("native_selected_frequency"),
            "v2_fallback": new["allfront_fallback_to_v1"],
            "full_search_cost_seconds": native["total_cpu_wall_seconds"],
        })
    effects_vs_v1 = np.array([
        float(row["v2_allfront_consensus_exact"]) -
        float(row["v1_selected_only_consensus_exact"])
        for row in entries], dtype=float)
    effects_vs_native = np.array([
        float(row["v2_allfront_consensus_exact"]) -
        float(row["native_pic_success_fraction"])
        for row in entries], dtype=float)
    p_v1, _ = _cluster_signflip_p(effects_vs_v1, seed=21011)
    p_native, _ = _cluster_signflip_p(effects_vs_native, seed=21012)
    return {
        "status": "COMPLETE_SECONDARY_EXPLORATORY_NOT_INDEPENDENT_CONFIRMATION",
        "manifest_sha": manifest["manifest_sha"],
        "discovery_code_sha": manifest["code_sha"],
        "unit_of_analysis": "physical_system",
        "n_independent_systems": len(entries),
        "n_evolutions": sum(r["n_full_searches"] for r in audited["systems"]),
        "n_ensemble_decisions": len(entries),
        "native_mean_single_search_success": audited["mean_native_single_search_success"],
        "v1_selected_only_consensus_success": float(np.mean(
            [x["v1_selected_only_consensus_exact"] for x in entries])),
        "v2_allfront_consensus_success": float(np.mean(
            [x["v2_allfront_consensus_exact"] for x in entries])),
        "v2_vs_v1_delta_pp": float(100 * np.mean(effects_vs_v1)),
        "v2_vs_native_delta_pp": float(100 * np.mean(effects_vs_native)),
        "v2_vs_v1_cluster_signflip_p": float(p_v1),
        "v2_vs_native_cluster_signflip_p": float(p_native),
        "inference_warning": (
            "Selection rule formulated while S2 jobs were already executing; "
            "secondary, small-N, nonconfirmatory. Four physical systems "
            "cannot achieve a two-sided sign-flip p below 0.125; "
            "K full evolutions are required per single decision."
        ),
        "entries": entries,
        "source_ledger": audited["ledger"],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    paths = sorted(p for p in args.records.rglob("*.json")
                   if len(p.stem) == 64 and
                   all(c in "0123456789abcdef" for c in p.stem))
    result = analyze_front_consensus(manifest, paths)
    args.output.mkdir(parents=True, exist_ok=True)
    json_path = args.output/"ALLFRONT_SECONDARY.json"
    tmp = json_path.with_suffix(".json.part")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n")
    tmp.replace(json_path)
    lines = [
        "# NIR1 secondary, truth-free all-Pareto-front ensemble",
        "",
        f"**Status:** {result['status']}",
        f"**Immutable run manifest SHA:** `{manifest['manifest_sha']}`",
        f"**Discovery SHA:** `{manifest['code_sha']}`",
        "",
        "This is an exploratory SECONDARY study. The algorithm was not "
        "fixed before the original S2 search began.",
        "",
        "| System | Native exact / 5 | V1 selected-only | V2 all-front | V2 modal runs |",
        "|---|---:|---:|---:|---:|"
    ]
    for row in result.get("entries", []):
        lines.append(
            f"| {row['system']} | "
            f"{row['native_pic_success_fraction'] * 5:.0f}/5 | "
            f"{int(row['v1_selected_only_consensus_exact'])} | "
            f"{int(row['v2_allfront_consensus_exact'])} | "
            f"{row['v2_support_run_frequency']} |"
        )
    if "v2_vs_native_delta_pp" in result:
        lines.extend([
            "", f"V2 minus native descriptive effect: {result['v2_vs_native_delta_pp']:+.1f} pp.",
            f"V2 minus frozen V1 effect: {result['v2_vs_v1_delta_pp']:+.1f} pp.",
            f"Independent-system exact two-sided sign-flip p against V1: "
            f"{result['v2_vs_v1_cluster_signflip_p']:.6g}.",
            "", result["inference_warning"],
        ])
    (args.output/"ALLFRONT_SECONDARY.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"status": result["status"],
                      "delta_pp": result.get("v2_vs_v1_delta_pp")}))
    return 0 if result["status"].startswith("COMPLETE") else 2


if __name__ == "__main__":
    raise SystemExit(main())
