"""E1 exact-first Gram reuse with cache integrity and reference fallback."""

from __future__ import annotations

import hashlib
import json
import time
from typing import MutableMapping, Sequence

import numpy as np

from .cache import gram_checksum
from .design import ResearchDesign
from .gram import GramBlocks, accumulate_gram, subset_gram
from .regularizers import SparseFit, fit_weighted_en


def make_cache_key(design: ResearchDesign, *, target_id: str, dtype: str,
                   revision: str) -> str:
    """Hash all *actual* derivative values, masks, weights and provenance."""
    h = hashlib.sha256()
    header = {"target_id": target_id, "dtype": dtype, "revision": revision,
              "token_names": design.token_names, "metadata": design.metadata}
    h.update(json.dumps(header, sort_keys=True, ensure_ascii=False,
                        allow_nan=False, default=str).encode())
    for array in (design.X, design.y, design.sample_weight, design.environment_id):
        arr = np.ascontiguousarray(array)
        h.update(str(arr.shape).encode())
        h.update(str(arr.dtype).encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def _expand_fit(fit: SparseFit, selected: np.ndarray, width: int,
                extra_reason: str | None) -> SparseFit:
    full_beta = np.zeros(width)
    full_support = np.zeros(width, dtype=bool)
    full_beta[selected] = fit.beta
    full_support[selected] = fit.support
    reasons = "; ".join(filter(None, [fit.fallback_reason, extra_reason])) or None
    return SparseFit(full_beta, full_support, fit.objective, fit.converged,
                     fit.kkt_residual, reasons, fit.iterations)


def fit_with_exact_cache(design: ResearchDesign, support: np.ndarray,
                         cache: MutableMapping[str, dict[str, object]], *,
                         l1: float, l2: float, weights: np.ndarray,
                         target_id: str = "target", revision: str = "nir1-v1") -> SparseFit:
    mask = np.asarray(support, dtype=bool)
    p = design.X.shape[1]
    if mask.shape != (p,) or not mask.any():
        raise ValueError("support must be a nonempty full-width boolean mask")
    idx = np.flatnonzero(mask)
    sub = ResearchDesign(design.X[:, idx], design.y, design.sample_weight,
                         design.environment_id,
                         tuple(design.token_names[j] for j in idx), design.metadata)
    key = make_cache_key(design, target_id=target_id, dtype=str(design.X.dtype),
                         revision=revision)
    entry = cache.get(key)
    extra_reason = None
    if entry is not None and ("gram" not in entry or
                              gram_checksum(entry["gram"]) != entry.get("checksum")):
        extra_reason = "cache_corrupt_recomputed"
        entry = None
    if entry is None:
        moments = accumulate_gram(design)
        entry = {"gram": moments, "checksum": gram_checksum(moments)}
        cache[key] = entry
    gram = subset_gram(entry["gram"], idx)

    # Cache reuse does not magically cure an ill-conditioned regression.
    # Fall back to the same reference arithmetic/selection on such inputs.
    colnorm = np.sqrt(np.maximum(np.diag(gram.G), 1e-300))
    standard = gram.G / np.outer(colnorm, colnorm)
    singular = np.linalg.svd(standard, compute_uv=False)
    bad_condition = bool(singular[-1] < 1e-10 * singular[0])
    if bad_condition:
        reference = fit_weighted_en(sub, l1=l1, l2=l2, weights=weights)
        return _expand_fit(reference, idx, p,
                           "; ".join(filter(None, [extra_reason, "reference_ill_conditioned"])))
    fast = fit_weighted_en(sub, l1=l1, l2=l2, weights=weights,
                           _precomputed_gram=gram)
    if not fast.converged or np.any((np.abs(fast.beta) > 0.) &
                                   (np.abs(fast.beta) < 1e-9)):
        reference = fit_weighted_en(sub, l1=l1, l2=l2, weights=weights)
        return _expand_fit(reference, idx, p,
                           "; ".join(filter(None, [extra_reason, "reference_near_threshold"])))
    return _expand_fit(fast, idx, p, extra_reason)


def compare_reference_fast(cases: Sequence[ResearchDesign], *, rtol: float = 1e-9,
                           atol: float = 1e-11) -> list[dict[str, object]]:
    cache: dict[str, dict[str, object]] = {}
    rows = []
    for case_idx, d in enumerate(cases):
        p = d.X.shape[1]
        mask = np.ones(p, dtype=bool)
        weights = np.ones(p)
        begin = time.perf_counter()
        reference = fit_weighted_en(d, l1=.01, l2=.01, weights=weights)
        reference_seconds = time.perf_counter() - begin
        begin = time.perf_counter()
        fast = fit_with_exact_cache(d, mask, cache, l1=.01, l2=.01,
                                    weights=weights, target_id=f"case{case_idx}")
        fast_seconds = time.perf_counter() - begin
        same_support = bool(np.array_equal(reference.support, fast.support))
        same_values = bool(np.allclose(reference.beta, fast.beta, rtol=rtol, atol=atol))
        same_objective = bool(np.isclose(reference.objective, fast.objective,
                                         rtol=rtol, atol=atol))
        rows.append({"case_index": case_idx, "n": d.X.shape[0], "p": p,
                     "support_equal": same_support,
                     "max_coefficient_abs_error": float(np.max(np.abs(reference.beta - fast.beta))),
                     "objective_abs_error": float(abs(reference.objective - fast.objective)),
                     "quality_changed": not (same_support and same_values and same_objective),
                     "reference_seconds": reference_seconds, "fast_seconds": fast_seconds,
                     "fallback": fast.fallback_reason, "approximate": False})
    return rows
