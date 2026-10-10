"""PIC dataset metadata and disjoint, predeclared system-level splits."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import yaml


def pic_dataset_names() -> tuple[str, ...]:
    from projects.pic.epde_bench.datasets import REGISTRY
    return tuple(sorted(REGISTRY))


def load_registered_split(path: Path | str, stage: str) -> tuple[str, ...]:
    parsed = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    groups = {key: tuple(parsed[key]) for key in ("development", "validation", "heldout")}
    unknown = set().union(*map(set, groups.values())) - set(pic_dataset_names())
    if unknown:
        raise ValueError(f"Unregistered PIC datasets: {sorted(unknown)}")
    flat = [name for values in groups.values() for name in values]
    if len(set(flat)) != len(flat):
        raise ValueError("System leakage: one dataset appears in multiple splits")
    if stage not in groups:
        raise ValueError(f"Unknown split: {stage}")
    return groups[stage]


def check_representability(dataset: str, tokens: Sequence[str]) -> dict[str, object]:
    """Conservative status until actual library/ground-truth compatibility is verified."""
    from projects.pic.epde_bench.datasets import REGISTRY
    spec = REGISTRY.get(dataset)
    if spec is None:
        return {"dataset": dataset, "status": "unsupported", "reason": "unknown dataset"}
    if spec.suite == "real":
        return {"dataset": dataset, "status": "unknown_truth",
                "reason": "measured-system truth is not guaranteed"}
    if not tokens:
        return {"dataset": dataset, "status": "unsupported", "reason": "empty library"}
    return {"dataset": dataset, "status": "not_verified",
            "reason": "verify actual EPDE pool factors and truth alternatives before a recovery score"}
