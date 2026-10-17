"""Series of runs over seeds and noise levels: success rates and the noise limit.

A series is several runs of one method on one record that differ only in the
seed (noise realisation and optimizer randomness) and the noise level. The
summary follows the PDEComp benchmark: a run succeeds when its selected
equation has exactly the terms of the known law (HD = 0); the coefficient error
is averaged over successful runs only; the noise limit is the highest tested
level up to which at least 3 of 5 runs (60 %) succeed at every level.
"""
from __future__ import annotations

import statistics

from epde_bench.metrics import wilson_ci

#: PDEComp's rule for the noise limit: at least 3 successes out of 5 runs.
LIMIT_RATE = 0.6
MIN_RUNS_FOR_LIMIT = 3


def _terminal(record):
    return isinstance(record, dict) and record.get("status") in ("ok", "error", "timeout")


def summarize(pairs, label_of):
    """``pairs``: (job, record) of one record; ``label_of(job)`` names the method.

    Returns rows per (method, noise) and the noise limit per method."""
    groups = {}
    for job, record in pairs:
        if not _terminal(record) or not (record.get("problem") or {}).get("truth"):
            continue
        key = (label_of(job), float(record.get("noise", job.get("noise", 0)) or 0))
        groups.setdefault(key, []).append(record)
    rows = []
    for (method, noise), records in sorted(groups.items()):
        ok = [r for r in records if r.get("status") == "ok"]
        metrics = [r.get("metrics") or {} for r in ok]
        wins = [m for m in metrics if m.get("success_selected")]
        low, high = wilson_ci(len(wins), len(records))
        hd = [m["hamming_selected"] for m in metrics if isinstance(m.get("hamming_selected"), int)]
        re = [m["re_sum"] for m in wins if isinstance(m.get("re_sum"), (int, float))]
        times = [r["fit_seconds"] for r in ok if isinstance(r.get("fit_seconds"), (int, float))]
        rows.append({"method": method, "noise, %": noise, "runs": len(records),
                     "found the law": len(wins), "success rate": len(wins) / len(records),
                     "95 % interval": f"{low:.0%} – {high:.0%}",
                     "HD, mean": statistics.fmean(hd) if hd else None,
                     "RE (sum), mean of successes": statistics.fmean(re) if re else None,
                     "failed runs": len(records) - len(ok),
                     "median time, s": statistics.median(times) if times else None})
    return rows, noise_limits(rows)


def noise_limits(rows):
    """``{method: highest noise level kept, or None}`` by the 3-of-5 rule."""
    by_method = {}
    for row in rows:
        by_method.setdefault(row["method"], []).append(row)
    limits = {}
    for method, items in by_method.items():
        limit = None
        for row in sorted(items, key=lambda r: r["noise, %"]):
            if row["runs"] < MIN_RUNS_FOR_LIMIT or row["success rate"] < LIMIT_RATE:
                break
            limit = row["noise, %"]
        limits[method] = limit
    return limits
