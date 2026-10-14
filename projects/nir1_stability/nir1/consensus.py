"""Truth-free cross-restart structural consensus for EPDE equation discovery.

One decision per SYSTEM from K separately seeded *full-budget* searches.
This is an ensemble with K times the computation of a single PIC run;
its accuracy must never be reported as K independent new successes.

The fixed rule uses only the native PIC-selected equation from each restart:
- require identical source code, experiment configuration, dataset, data_seed;
- canonicalize structure ignoring coefficient magnitudes and target orientation;
- choose a unique most-frequent structure with frequency >= 2;
- if all structures unique or the top mode ties, return the lowest-seed
  PIC-selected equation (predeclared deterministic fallback).
Truth and experimental evaluation labels are never inspected.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Sequence

from projects.pic.epde_bench.metrics import canonical_tokens


def _signature(texts: Sequence[str]) -> str:
    """Stable JSON structural key; robust to Python frozenset repr ordering."""
    system = canonical_tokens(texts)
    equations = []
    for equation in system:
        terms = []
        for term in equation:
            factors = []
            for label, params in term:
                factors.append([str(label), [[str(k), repr(v)] for k, v
                                             in sorted(params, key=lambda item: str(item[0]))]])
            terms.append(sorted(factors, key=lambda pair: json.dumps(pair, sort_keys=True)))
        equations.append(sorted(terms, key=lambda item: json.dumps(item, sort_keys=True)))
    return json.dumps(sorted(equations, key=lambda item: json.dumps(item, sort_keys=True)),
                      ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def consensus_decision(records: Sequence[dict], *, min_repetition: int = 2) -> dict:
    """Return one chosen equation with provenance, WITHOUT consulting truth.

    records are untouched PIC results, not the outer campaign artifacts.
    A failed search cannot simply be removed: fail the ensemble and report
    it as incomplete until the scientific analysis handles the failure.
    """
    if not records or min_repetition < 2:
        raise ValueError("At least one record and min_repetition >= 2 required")
    if any(rec.get("status") != "ok" for rec in records):
        raise ValueError("Failed or missing full EPDE search; never silently omit it")
    ordered = sorted(records, key=lambda rec: int(rec["seed"]))
    seeds = [int(rec["seed"]) for rec in ordered]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Repeated optimizer seeds are not independent restarts")
    for field in ("dataset", "noise", "search_config"):
        control = json.dumps(ordered[0].get(field), sort_keys=True)
        if any(json.dumps(row.get(field), sort_keys=True) != control for row in ordered[1:]):
            raise ValueError(f"Nonidentical {field}; cannot vote across different problems")
    # Data seed must be frozen independently of the optimizer RNG.
    def data_seed(row):
        return int(row.get("config", {}).get("nir1_data_seed", row.get("seed")))
    if len({data_seed(row) for row in ordered}) != 1:
        raise ValueError("Different data realizations cannot be merged into one ensemble")
    revisions = {str(row.get("environment", {}).get("epde_commit")) for row in ordered}
    if len(revisions) != 1 or "None" in revisions:
        raise ValueError("Mixed or missing source revisions")
    chosen = []
    for row in ordered:
        candidates = row.get("front") or []
        idx = (row.get("metrics") or {}).get("selected_index")
        if not isinstance(idx, int) or not 0 <= idx < len(candidates):
            raise ValueError("Missing or invalid native PIC-selected front index")
        texts = candidates[idx]
        chosen.append({"seed": int(row["seed"]), "key": _signature(texts),
                       "texts": texts})
    freq = Counter(item["key"] for item in chosen)
    ranked = freq.most_common()
    highest = ranked[0][1]
    winners = [key for key, count in freq.items() if count == highest]
    use_mode = highest >= min_repetition and len(winners) == 1
    winner = (next(item for item in chosen if item["key"] == winners[0])
              if use_mode else chosen[0])
    return {
        "method": "nir1_cross_restart_consensus_v1",
        "n_full_searches": len(ordered), "source_revisions": sorted(revisions),
        "dataset": ordered[0]["dataset"], "data_seed": data_seed(ordered[0]),
        "optimizer_seeds": seeds, "mode_frequency": highest,
        "mode_used": bool(use_mode), "tie_or_no_repeat": not use_mode,
        "chosen_seed": winner["seed"], "chosen_equations": winner["texts"],
        "chosen_structure_sha256": hashlib.sha256(winner["key"].encode()).hexdigest(),
        "distinct_structures": len(freq), "all_seeds_accounted": True,
        "truth_used_to_select": False,
        "fidelity": "one output per system and data realization; K complete EPDE evolutions",
    }


def min_discrepancy_restart(records: Sequence[dict]) -> dict:
    """Equal-compute K-search control: pick native PIC result with min loss.

    All K full EPDE searches are still paid for. This is a truth-free
    competing way of deriving ONE output from K restarts; unlike comparing
    consensus to average per-seed PIC success, this control has the SAME
    number of searches and same single-decision output unit.
    """
    import math

    prior = consensus_decision(records)  # independent-seed / code / config gate
    scored = []
    for row in records:
        index = int(row["metrics"]["selected_index"])
        if not isinstance(row.get("objectives"), list) or not 0 <= index < len(row["objectives"]):
            raise ValueError("Cannot compare selected loss across missing fronts")
        objective = row["objectives"][index]
        if (not isinstance(objective, list) or not objective or
                len(objective) != 2 * len(row["front"][index])):
            raise ValueError("Malformed objective for one equation per axis pair")
        vals = [float(v) for v in objective]
        if any(not math.isfinite(v) for v in vals):
            raise ValueError("Nonfinite objective cannot be used for cross-restart selection")
        scored.append((sum(vals[::2])/len(row["front"][index]), int(row["seed"]), index, row))
    loss, seed, index, selected = min(scored, key=lambda item: (item[0], item[1]))
    eq = selected["front"][index]
    key = _signature(eq)
    return {"method": "nir1_equal_compute_lowest_native_discrepancy_v1",
            "dataset": prior["dataset"], "data_seed": prior["data_seed"],
            "optimizer_seeds": prior["optimizer_seeds"],
            "n_full_searches": len(records), "source_revisions": prior["source_revisions"],
            "chosen_seed": seed, "chosen_equations": eq,
            "chosen_structure_sha256": hashlib.sha256(key.encode()).hexdigest(),
            "mean_native_discrepancy": float(loss), "truth_used_to_select": False,
            "fidelity": "one output from K standard PIC full searches, budget matched to consensus"}
