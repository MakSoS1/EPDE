"""Failure-aware denominators and paired, system-clustered uncertainty."""

from __future__ import annotations

from collections import defaultdict
from itertools import product
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import binomtest, norm

from .records import STATUSES


def wilson_interval(successes: int, n: int, confidence: float = .95) -> tuple[float, float]:
    if n <= 0 or not 0 <= successes <= n or not 0 < confidence < 1:
        raise ValueError("Invalid successes, denominator or confidence")
    z = float(norm.ppf(.5 + confidence / 2))
    p = successes / n
    divisor = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / divisor
    radius = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / divisor
    return float(max(center - radius, 0.)), float(min(center + radius, 1.))


def summarize_outcomes(ledger: Mapping[str, object],
                       records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Never use only successful rows as the unlabeled denominator."""
    planned = int(ledger["planned"])
    counts = {status: int(ledger["status_counts"].get(status, 0)) for status in STATUSES}
    if any(count < 0 for count in counts.values()) or sum(counts.values()) != planned:
        raise ValueError("Status totals do not reconcile with immutable planned count")
    eligible = planned - counts["unsupported"]
    rate = counts["ok"] / eligible if eligible else None
    conclusion = "NOT RUN" if planned == 0 else ("PROVISIONAL" if counts["incomplete"] else "RECORDED")
    return {"total_planned": planned, "supported_planned": eligible,
            "supported_success_rate": rate, "conditional_on_ok": counts["ok"],
            "status_counts": counts, "complete": counts["incomplete"] == 0,
            "conclusion_status": conclusion}


def _cluster_signflip_p(effects: np.ndarray, *, seed: int) -> tuple[float, str]:
    """Two-sided paired sign-flip on independent SYSTEM effects, not run seeds.

    Exact enumeration is used for at most 18 systems; beyond this limit the
    fixed-seed Monte Carlo p-value includes the observed arrangement (+1).
    """
    values = np.asarray(effects, dtype=float)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("Expected nonempty finite system effects")
    observed = abs(float(values.mean()))
    cutoff = observed - max(1e-12, 1e-12 * observed)
    if len(values) <= 18:
        extreme = 0
        for signs in product((-1., 1.), repeat=len(values)):
            extreme += abs(float(np.dot(values, signs)) / len(values)) >= cutoff
        return float(extreme / (1 << len(values))), "exact_cluster_signflip"
    rng = np.random.default_rng(seed)
    total = 99999
    extreme = 0
    for _ in range(total):
        signs = rng.choice((-1., 1.), size=len(values))
        extreme += abs(float(np.dot(values, signs)) / len(values)) >= cutoff
    return float((extreme + 1) / (total + 1)), "monte_carlo_cluster_signflip"


def paired_effect_ci(a: Sequence[bool], b: Sequence[bool], groups: Sequence[str], *,
                     n_boot: int = 2000, seed: int = 0) -> dict[str, float | int | str]:
    """Method B minus A, each independent physical system weighted equally.

    Repeated optimizer seeds within a system do NOT contribute independent
    degrees of freedom to p-values or cluster bootstrap uncertainty.
    """
    if not len(a) or not (len(a) == len(b) == len(groups)) or n_boot < 100:
        raise ValueError("Paired arrays must be nonempty and equal in length; >=100 bootstrap draws")
    a_arr, b_arr = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    cluster_ids: dict[str, list[int]] = defaultdict(list)
    for i, group in enumerate(groups):
        cluster_ids[str(group)].append(i)
    cluster_names = sorted(cluster_ids)
    effects = np.asarray([
        float(np.mean(b_arr[cluster_ids[name]].astype(float) -
                      a_arr[cluster_ids[name]].astype(float)))
        for name in cluster_names
    ])
    delta = float(100 * effects.mean())
    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot)
    for iteration in range(n_boot):
        selected = rng.integers(0, len(effects), size=len(effects))
        samples[iteration] = float(100 * effects[selected].mean())
    low, high = (float(v) for v in np.quantile(samples, [.025, .975]))
    a_only = int(np.sum(a_arr & ~b_arr))
    b_only = int(np.sum(~a_arr & b_arr))
    discordant = a_only + b_only
    # This pooled McNemar value is diagnostic ONLY: it treats optimizer seeds
    # within one system as independent and MUST NOT be used for inference.
    pooled_diagnostic_p = (float(binomtest(min(a_only, b_only), discordant, .5).pvalue)
                           if discordant else 1.)
    p_value, p_method = _cluster_signflip_p(effects, seed=seed)
    return {"delta_pp": delta, "ci_low_pp": low, "ci_high_pp": high,
            "n_pairs": len(a), "n_clusters": len(cluster_names),
            "discordant": discordant, "a_only": a_only, "b_only": b_only,
            "cluster_signflip_p": p_value, "cluster_signflip_method": p_method,
            "effect_unit": "independent_system_equal_weight",
            "mcnemar_exact_p_diagnostic_only": pooled_diagnostic_p}


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Step-down family-wise multiplicity correction."""
    n = len(p_values)
    result = [1.] * n
    previous = 0.
    for rank, index in enumerate(sorted(range(n), key=lambda i: p_values[i])):
        adjusted = min(1., float(p_values[index]) * (n - rank))
        previous = max(previous, adjusted)
        result[index] = previous
    return result
