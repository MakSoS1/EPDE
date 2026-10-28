"""Versioned interoperability contract for EPDE, PySINDy (as run by PDEComp) and an external LLM.

Native EPDE records are kept intact and augmented with explicit criterion metadata.
Different methods do NOT share a numerical Pareto objective unless their
criterion definitions, normalizations and preprocessing are proven equal.
"""
from __future__ import annotations

import math
from typing import Any

SCHEMA_VERSION = "discovery-demo/1.0"
METHODS = ("epde", "pdecomp", "deepmod", "discover", "llm")
MAX_CANDIDATES = 512  # full native Pareto front; transport remains size-limited


def criterion_labels(method: str, record: dict) -> tuple[str, str]:
    if method == "epde":
        cfg = record.get("search_config") or (record.get("config") or {}).get("search", {})
        obj = cfg.get("objectives") or {}
        return (str(obj.get("discrepancy_metric", "discrepancy")),
                str(obj.get("second_objective", "EPDE second objective")))
    if method in ("pdecomp", "deepmod", "discover"):
        return ("not reported by PDEComp", "not reported by PDEComp")
    return ("LLM criterion 1 (provider-defined)", "LLM criterion 2 (provider-defined)")


def normalize(record: dict, method: str) -> dict:
    """Check a method's response without fabricating Pareto candidates or scores."""
    if not isinstance(record, dict) or method not in METHODS:
        raise ValueError("Unknown method or response type")
    status = record.get("status")
    if status not in ("ok", "unsupported", "error", "timeout", "stopped"):
        raise ValueError("Invalid result status")
    out = dict(record)
    out["schema_version"] = SCHEMA_VERSION
    out["method"] = method
    if status != "ok":
        out.setdefault("front", [])
        out.setdefault("objectives", [])
        return out
    front, objectives = out.get("front"), out.get("objectives")
    if not isinstance(front, list) or not isinstance(objectives, list):
        raise ValueError("Missing candidate list or objective list")
    if len(front) != len(objectives) or len(front) > MAX_CANDIDATES:
        raise ValueError("Candidate/objective length mismatch or too many candidates")
    for candidate, objective in zip(front, objectives):
        if not isinstance(candidate, list) or not 1 <= len(candidate) <= 10:
            raise ValueError("Each candidate must contain 1-10 equations")
        if any(not isinstance(eq, str) or not 1 <= len(eq) <= 4096 or "=" not in eq
               for eq in candidate):
            raise ValueError("Candidate equation must be a short string containing '='")
        if objective is not None and (not isinstance(objective, (tuple, list))
                                      or len(objective) < 2
                                      or not all(isinstance(v, (int, float)) and math.isfinite(v)
                                                 for v in objective[:2])):
            raise ValueError("Objective must have two finite numerical values")
    metrics = out.get("metrics") or {}
    if not isinstance(metrics, dict):
        raise ValueError("Invalid metrics")
    index = metrics.get("selected_index")
    if index is not None and (type(index) is not int or not 0 <= index < len(front)):
        raise ValueError("Invalid selected_index")
    if index is not None and out.get("selected") != front[index]:
        raise ValueError("Selected equation does not match selected_index")
    labels = out.get("criterion_labels")
    if labels is None:
        labels = criterion_labels(method, out)
    if not (isinstance(labels, (list, tuple)) and len(labels) == 2
            and all(isinstance(label, str) and 1 <= len(label) <= 120 for label in labels)):
        raise ValueError("Exactly two criterion labels are required")
    out["criterion_labels"] = list(labels)
    out["objectives_comparable_between_methods"] = False
    return out


def llm_payload_valid(reply: Any) -> dict:
    """Yaroslav's private service uses the same record schema; never execute code."""
    reply = normalize(reply, "llm")
    # Service must specify what it optimized: invented or unlabeled axes mislead.
    if reply.get("status") == "ok" and reply["criterion_labels"] == list(criterion_labels("llm", {})):
        raise ValueError("LLM service must specify its two measured criteria")
    return reply
