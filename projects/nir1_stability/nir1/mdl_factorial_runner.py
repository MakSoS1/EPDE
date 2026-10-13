"""Frozen causal ablation of H9/H10/H11/H12, no modification of EPDE defaults.

Arms:
 native              unrestricted PIC baseline (reference)
 anchor_utopia       H9 only, default normalized compromise
 anchor_min          H9, min first fitness (NO MDL)
 mdl_utopia          H9+H10, default normalized compromise
 mdl_min             H9+H10+H11, no repair
 mdl_min_repair      H9+H10+H11+H12
 no_mdl_min_repair   H9+H11+H12 (isolates causal contribution of H10)

Known arms delegate to the existing 2026-10-11 research adapter.
The two no-MDL arms are direct wrappers around the IDENTICAL PIC run_one
with exactly the same flags, prior, data, token pool and search overrides.
The selector is applied BEFORE access to simulator ground truth.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path


FROZEN_ARMS = {
    "native": "default",
    "anchor_utopia": "nir1_dynamic_anchor",
    "anchor_min": None,
    "mdl_utopia": "nir1_dynamic_mdl",
    "mdl_min": "nir1_dynamic_mdl_select",
    "mdl_min_repair": "nir1_dynamic_mdl_repair",
    "no_mdl_min_repair": None,
}
ENV_KEYS = ("NIR1_DYNAMIC_TARGET", "NIR1_DYNAMIC_MDL", "NIR1_DYNAMIC_REPAIR")


def _select_without_truth(record: dict, dataset: str) -> dict:
    """Choose a Pareto index on objectives, THEN evaluate against truth."""
    from projects.nir1_stability.nir1.causal_mdl_selector import (
        select_min_causal_description,
    )

    objective_vectors = record.get("objectives") or []
    chosen = select_min_causal_description(objective_vectors)
    record["metrics_pic_original"] = copy.deepcopy(record.get("metrics", {}))
    record["research_selected_index"] = chosen
    record["research_selected"] = (
        record["front"][chosen] if chosen is not None else None
    )
    record["selected"] = record["research_selected"]
    record["selection_strategy"] = "argmin_first_objective_without_mdl"

    from projects.pic.epde_bench.datasets import load
    from projects.pic.epde_bench.metrics import canonical_tokens, hamming_best

    accepted = load(dataset, **record.get("config", {}).get("loader", {})).truth_systems
    score = record["metrics"]
    score["selected_index"] = chosen
    if accepted:
        truth = [canonical_tokens(eq) for eq in accepted]
        dist = (
            hamming_best(canonical_tokens(record["front"][chosen]), truth)
            if chosen is not None else None
        )
        score["success_selected"] = bool(dist == 0) if dist is not None else False
        score["hamming_selected"] = dist
    return record


def _run_unpenalized(
    dataset: str, arm: str, seed: int, data_seed: int,
    *, noise: float, smoke: bool,
) -> dict:
    """Use real EPDE with temporal anchoring, optionally H12 repair, no MDL."""
    from projects.pic.epde_bench.config import deep_merge
    from projects.pic.epde_bench.runner import run_one

    overrides = {"search": {"objectives": {
        "instability_metric": "chi2", "sparsity_cls": "vwsr",
    }}}
    if smoke:
        overrides = deep_merge(overrides, {"search": {"evolution": {
            "population_size": 4, "training_epochs": 1,
        }}})
    overrides["nir1_data_seed"] = data_seed
    old = {k: os.environ.get(k) for k in ENV_KEYS}
    try:
        os.environ["NIR1_DYNAMIC_TARGET"] = "1"
        os.environ.pop("NIR1_DYNAMIC_MDL", None)
        if arm == "no_mdl_min_repair":
            os.environ["NIR1_DYNAMIC_REPAIR"] = "1"
        else:
            os.environ.pop("NIR1_DYNAMIC_REPAIR", None)
        record = run_one(dataset, "default", noise, seed, overrides=overrides)
    finally:
        for k, value in old.items():
            if value is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = value
    record["research_variant"] = arm
    record["research_overrides"] = overrides
    record["research_physical_prior"] = (
        "Bare time derivative target for explained variable, existing token pool; "
        "MDL disabled; " + ("H12 temporal repair enabled" if arm == "no_mdl_min_repair"
        else "H12 temporal repair disabled")
    )
    if record.get("status") == "ok":
        record = _select_without_truth(record, dataset)
    return record


def execute(dataset: str, arm: str, seed: int, data_seed: int,
            *, noise: float = 0.0, smoke: bool = False) -> dict:
    if arm not in FROZEN_ARMS:
        raise ValueError(f"Unknown frozen arm {arm!r}")
    if FROZEN_ARMS[arm] is None:
        record = _run_unpenalized(dataset, arm, seed, data_seed,
                                  noise=noise, smoke=smoke)
    else:
        from projects.nir1_stability.nir1.epde_adapter import run_nir1_epde
        overrides = (
            {"search": {"evolution": {"population_size": 4,
                                      "training_epochs": 1}}}
            if smoke else None
        )
        record = run_nir1_epde(
            dataset, FROZEN_ARMS[arm], seed, noise,
            data_seed=data_seed, overrides=overrides,
        )
    record["factorial_arm"] = arm
    record["optimizer_seed"] = seed
    record["data_seed_requested"] = data_seed
    record["nir1_smoke_only"] = smoke
    record["factorial_protocol"] = "mdlablation20261014-v1"
    return record


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=".factorial-", delete=False,
        ) as stream:
            staged = stream.name
            json.dump(payload, stream, allow_nan=False, sort_keys=True,
                      indent=2, default=str)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staged, path)
    finally:
        if staged and os.path.exists(staged):
            os.unlink(staged)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--arm", choices=tuple(FROZEN_ARMS), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--data-seed", type=int, default=0)
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    try:
        record = execute(args.dataset, args.arm, args.seed, args.data_seed,
                         noise=args.noise, smoke=args.smoke)
    except Exception as exc:
        record = {
            "status": "crash",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "dataset": args.dataset,
            "factorial_arm": args.arm,
            "optimizer_seed": args.seed,
            "data_seed_requested": args.data_seed,
            "nir1_smoke_only": args.smoke,
            "factorial_protocol": "mdlablation20261014-v1",
        }
    _atomic_json(args.output, record)
    print(json.dumps({
        "status": record.get("status"), "dataset": args.dataset,
        "arm": args.arm, "seed": args.seed, "smoke": args.smoke,
        "output": str(args.output),
    }), flush=True)
    return 0 if record.get("status") == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
