"""Replay NIR1-v2 sparsefront on original S1 search fronts without retraining.

This is DEVELOPMENT analysis on data used to select complexity coefficients,
NOT independent confirmation. A separate Actions S1b campaign is required.
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path

from projects.pic.epde_bench.metrics import canonical_tokens, hamming_best
from .selectors import select_sparsefront


def replay(evidence_archive: Path) -> dict:
    rows = []
    with zipfile.ZipFile(evidence_archive) as outer:
        shards = [name for name in outer.namelist()
                  if name.startswith("nir1-s1-shard") and name.endswith(".zip")]
        if len(shards) != 4:
            raise ValueError("Expected exactly four original S1 shards")
        for shard in shards:
            with zipfile.ZipFile(io.BytesIO(outer.read(shard))) as inner:
                for filename in inner.namelist():
                    if not filename.endswith(".json"):
                        continue
                    record = json.loads(inner.read(filename))
                    if record["status"] != "ok":
                        raise ValueError("Cannot omit failed S1 records")
                    run = record["rows"][0]
                    if run["research_variant"] != "default":
                        continue
                    if run["problem"]["n_truth_alternatives"]:
                        # Do not silently discard accepted alternative forms.
                        from projects.pic.epde_bench.datasets import load
                        truth = load(run["dataset"]).truth_systems
                    else:
                        truth = [run["problem"]["truth"]]
                    canonical_truth = [canonical_tokens(equations) for equations in truth]
                    chosen = select_sparsefront(run["front"], run["objectives"])
                    predicted = (chosen is not None and hamming_best(
                        canonical_tokens(run["front"][chosen]), canonical_truth) == 0)
                    rows.append({"system": run["dataset"], "seed": run["seed"],
                                 "PIC_success": bool(run["metrics"]["success_selected"]),
                                 "sparsefront_success": bool(predicted),
                                 "sparsefront_selected_index": chosen,
                                 "PIC_selected_index": run["metrics"]["selected_index"]})
    if len(rows) != 30 or len({(r["system"], r["seed"]) for r in rows}) != 30:
        raise ValueError("Retrospective S1 data incomplete; refuse cherry-picking")
    groups = defaultdict(list)
    for row in rows:
        groups[row["system"]].append(row)
    return {
        "source": str(evidence_archive), "stage": "S1_RETROSPECTIVE_DEVELOPMENT",
        "independent_confirmation": False,
        "planned": 30, "replayed": len(rows),
        "PIC_selected_exact": sum(int(r["PIC_success"]) for r in rows),
        "sparsefront_selected_exact": sum(int(r["sparsefront_success"]) for r in rows),
        "per_system": {name: {
            "PIC": sum(int(r["PIC_success"]) for r in rr),
            "sparsefront": sum(int(r["sparsefront_success"]) for r in rr),
            "n": len(rr)} for name, rr in sorted(groups.items())},
        "rows": sorted(rows, key=lambda r: (r["system"], r["seed"])),
        "warning": ("S1 was used for selecting the selector weights; "
                    "result is descriptive and cannot validate an improvement"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(replay(args.evidence), ensure_ascii=False,
                                   sort_keys=True, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
