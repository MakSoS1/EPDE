"""Replay alternative final selectors on the SAME immutable EPDE Pareto front.

This saves one whole EPDE evolution for a selector-only comparison. It is
NOT valid for comparing regularizers or new Pareto objective functions.
Ground truth is used strictly after both candidates are selected.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from .selectors import select_sparsefront


def _canonical_json(body):
    return json.dumps(body, sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(",", ":")).encode("utf-8")


def paired_selector_replay(native_record: dict, *, truth_systems) -> dict:
    """Data-only selector followed by optional offline truth scoring.

    Requires an authentic PIC EPDE record. No new evolution or regularization.
    Preserves the original PIC verdict and all immutable candidate values.
    """
    from projects.pic.epde_bench.metrics import canonical_tokens, hamming_best

    if native_record.get("status") != "ok":
        raise ValueError("Only a completed successful EPDE front can be replayed")
    original = copy.deepcopy(native_record)
    front = original["front"]
    objectives = original["objectives"]
    old = copy.deepcopy(original["metrics"])
    new_index = select_sparsefront(front, objectives)
    if new_index is None:
        raise ValueError("No eligible finite Pareto candidate")
    if old.get("selected_index") is None:
        raise ValueError("Original PIC selector is missing, cannot pair")
    old_index = int(old["selected_index"])
    if not 0 <= old_index < len(front):
        raise ValueError("Original PIC selector index is out of bounds")
    distances = None
    if truth_systems:
        alts = [canonical_tokens(system) for system in truth_systems]
        distances = [hamming_best(canonical_tokens(candidate), alts)
                     for candidate in front]
    report = {
        "record_type": "PAIRED_POSTPROCESSOR_NO_NEW_EPDE_RUN",
        "source_front_sha256": hashlib.sha256(_canonical_json({
            "front": front, "objectives": objectives})).hexdigest(),
        "original_epde_code_sha": original.get("environment", {}).get("epde_commit"),
        "system": original.get("dataset"),
        "optimizer_seed": original.get("seed"),
        "same_front_for_both": True,
        "evolutions_per_pair": 1,
        "native_pic": {"selected_index": old_index,
                       "success_selected": old.get("success_selected"),
                       "hamming_selected": old.get("hamming_selected")},
        "sparsefront_v2": {"selected_index": new_index,
                           "success_selected": (distances[new_index] == 0
                                                if distances is not None else None),
                           "hamming_selected": (distances[new_index]
                                                if distances is not None else None)},
    }
    # No truth used in selection; it is neither returned nor hashed.
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Zero-extra-evolution Pareto selector replay")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--truth-dataset", default=None,
                        help="Use known PIC truth offline for scoring only")
    args = parser.parse_args(argv)
    record = json.loads(args.input.read_text(encoding="utf-8"))
    systems = None
    if args.truth_dataset is not None:
        if args.truth_dataset != record.get("dataset"):
            parser.error("Truth dataset must match recorded EPDE dataset")
        from projects.pic.epde_bench.datasets import load
        systems = load(args.truth_dataset, **record.get("config", {}).get("loader", {})).truth_systems
    body = paired_selector_replay(record, truth_systems=systems)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".part")
    tmp.write_bytes(_canonical_json(body) + b"\n")
    tmp.replace(args.output)
    print(json.dumps({"same_front": True, "chosen": body["sparsefront_v2"]["selected_index"],
                      "source_front_sha256": body["source_front_sha256"]}))


if __name__ == "__main__":
    main()
