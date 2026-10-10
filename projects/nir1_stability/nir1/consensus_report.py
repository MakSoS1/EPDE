"""Immutable ensemble analysis: one output per physical system, no seed pseudoreplication."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .campaign import resume_plan
from .consensus import consensus_decision
from .metrics import _cluster_signflip_p
from .records import read_record


def analyze_consensus(manifest: dict, records: list[Path]) -> dict:
    if manifest.get("stage") not in {"S1", "S2", "S3"}:
        raise ValueError("Ensemble analysis requires full EPDE campaign")
    methods = {str(run["method"]) for run in manifest["runs"]}
    if methods != {"default"}:
        raise ValueError("Consensus replay must receive ONE native PIC baseline search per seed")
    ledger = resume_plan(manifest, records)
    if ledger["invalid_records"] or ledger["remaining"] or (
            ledger["status_counts"]["ok"] != ledger["planned"]):
        return {"status": "INCOMPLETE_OR_FAILURE_BLOCKS_CONFIRMATION",
                "manifest_sha": manifest["manifest_sha"], "ledger": ledger}
    by_id = {}
    for path in records:
        rec = read_record(path)
        if rec["run_id"] in by_id:
            raise ValueError("Duplicate experiment run_id in ensemble input")
        by_id[rec["run_id"]] = rec
    by_system = defaultdict(list)
    for run in manifest["runs"]:
        outer = by_id[run["run_id"]]
        if outer.get("manifest_sha") != manifest["manifest_sha"] or outer["status"] != "ok":
            raise ValueError("Mismatched or failed primary evidence")
        row = outer["rows"][0]
        if row.get("research_variant") != "default":
            raise ValueError("A non-baseline full EPDE variant was supplied")
        if row.get("environment", {}).get("epde_commit") != manifest["code_sha"][:7]:
            raise ValueError("Wrong EPDE source revision")
        if row.get("seed") != run["optimizer_seed"] or row.get("dataset") != run["case"]:
            raise ValueError("Run identity and child EPDE record differ")
        by_system[(str(run["case"]), int(run["data_seed"]))].append(row)
    entries = []
    for (case, data_seed), runs in sorted(by_system.items()):
        choice = consensus_decision(runs)
        matched = next(row for row in runs if row["seed"] == choice["chosen_seed"])
        label = (matched.get("metrics") or {}).get("success_selected")
        if not isinstance(label, bool):
            raise ValueError("No precomputed offline PIC truth verdict for selected support")
        native_successes = sum(int(row["metrics"].get("success_selected") is True)
                               for row in runs)
        cost = sum(float(row["total_seconds"]) for row in runs)
        entries.append({
            "system": case, "data_seed": data_seed, "n_full_searches": len(runs),
            "baseline_selected_exact": native_successes,
            "baseline_rate": native_successes / len(runs),
            "consensus_selected_exact": label, "chosen_seed": choice["chosen_seed"],
            "modal_frequency": choice["mode_frequency"],
            "distinct_structures": choice["distinct_structures"],
            "mode_used": choice["mode_used"], "ties_fallback": choice["tie_or_no_repeat"],
            "total_cpu_wall_seconds": cost, "selected_structure_sha256":
                choice["chosen_structure_sha256"]})
    effects = np.asarray([float(row["consensus_selected_exact"])-row["baseline_rate"]
                          for row in entries], dtype=float)
    probability, method = _cluster_signflip_p(effects, seed=1026)
    return {
        "status": "COMPLETE_EXPLORATORY" if manifest["stage"] == "S1" else "COMPLETE_PROSPECTIVE_SMALL_N",
        "manifest_sha": manifest["manifest_sha"],
        "source_sha": manifest["code_sha"], "stage": manifest["stage"],
        "ledger": ledger, "n_independent_systems": len(entries),
        "n_total_full_searches": sum(x["n_full_searches"] for x in entries),
        "mean_native_single_search_success": float(np.mean([x["baseline_rate"] for x in entries])),
        "consensus_system_success": float(np.mean([x["consensus_selected_exact"] for x in entries])),
        "difference_pp_descriptive": float(100*np.mean(effects)),
        "exact_system_cluster_signflip_p": probability, "p_method": method,
        "fairness_note": "Consensus uses all K full searches to output ONE equation. Native baseline produces K different decisions; normalized rates differ in cost per decision. Comparison is descriptive, NOT equal-budget single-run superiority.",
        "systems": entries,
    }


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--records",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    opt=parser.parse_args(argv)
    manifest=json.loads(opt.manifest.read_text(encoding="utf-8"))
    files=sorted(p for p in opt.records.rglob("*.json")
                 if len(p.stem)==64 and all(c in '0123456789abcdef' for c in p.stem))
    result=analyze_consensus(manifest,files)
    opt.output.mkdir(parents=True,exist_ok=True)
    out=opt.output/"CONSENSUS_SUMMARY.json"
    temp=out.with_suffix(".json.part")
    temp.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n",
                    encoding="utf-8")
    temp.replace(out)
    lines=[
        "# NIR1 truth-free structural consensus, evidence report",
        "", f"Status: **{result['status']}**.",
        f"Code SHA: `{manifest['code_sha']}`.",
        f"Manifest SHA: `{manifest['manifest_sha']}`.",
        "", "This is ONE decision from K fully executed searches, NOT K independent predictions.",
        "", "| System | K | PIC exact among K | Consensus exact | Mode frequency | Total CPU seconds |",
        "|---|---:|---:|---|---:|---:|"]
    for row in result.get("systems",[]):
        lines.append(f"| {row['system']} | {row['n_full_searches']} | "
                     f"{row['baseline_selected_exact']}/{row['n_full_searches']} | "
                     f"{'yes' if row['consensus_selected_exact'] else 'no'} | "
                     f"{row['modal_frequency']} | {row['total_cpu_wall_seconds']:.1f} |")
    if "difference_pp_descriptive" in result:
        lines += ["",f"Descriptive effect: {result['difference_pp_descriptive']:+.1f} percentage points.",
                  f"Exact independent-system sign-flip p = {result['exact_system_cluster_signflip_p']:.6g}.",
                  "",result["fairness_note"]]
    (opt.output/"CONSENSUS_SUMMARY.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"status":result["status"],"n":result.get("n_total_full_searches",0),
                      "difference_pp":result.get("difference_pp_descriptive")}))
    return 0 if result["status"].startswith("COMPLETE") else 2


if __name__=="__main__":
    raise SystemExit(main())
