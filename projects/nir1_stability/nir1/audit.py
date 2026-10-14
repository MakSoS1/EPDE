"""S0 frozen-candidate science with oracle data used only for evaluation."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from .design import ResearchDesign
from .diagnostics import estimate_diagnostics
from .fixtures import make_fixture
from .historical import historical_vclog_weights
from .identity import canonical_run_id, hash_file
from .model_scores import score_equation
from .regularizers import build_penalty_weights, fit_weighted_en


METHODS = ("baseline_ols", "lasso", "instability", "identifiability", "hybrid",
           "historical_proxy")


def score_fixed_candidate(design: ResearchDesign, *, criterion: str = "nir1_excess") -> float:
    """Evaluate predeclared criterion without access to external oracle truth."""
    diag = estimate_diagnostics(design)
    if criterion == "residual":
        w = design.sample_weight
        residual = design.y - design.X @ diag.beta
        return float(np.dot(w, residual**2) / max(np.dot(w, design.y**2), 1e-30))
    if criterion == "historical_proxy":
        weights, _ = historical_vclog_weights(design, diag)
        return float(np.mean(weights))
    if criterion == "nir1_excess":
        return score_equation(diag)
    if criterion == "nir1_identifiability":
        base = score_equation(diag)
        return float(base + .25 * np.mean(1. - diag.identifiability))
    raise ValueError(f"Unknown candidate criterion {criterion}")


def _candidate_masks(truth: dict[str, object], p: int) -> dict[str, np.ndarray]:
    """Oracle constructs *external benchmark candidates*, not search choices."""
    known_support = np.zeros(p, dtype=bool)
    known_support[np.asarray(truth["support"], dtype=int)] = True
    masks = {"truth": known_support}
    if known_support.sum() > 1:
        missing = known_support.copy()
        missing[np.flatnonzero(missing)[-1]] = False
        masks["missing_term_decoy"] = missing
    unused = np.flatnonzero(~known_support)
    if len(unused):
        added = known_support.copy()
        added[unused[0]] = True
        masks["extra_term_decoy"] = added
        swapped = known_support.copy()
        swapped[np.flatnonzero(known_support)[0]] = False
        swapped[unused[0]] = True
        masks["replacement_decoy"] = swapped
    return masks


def _method_weights(design: ResearchDesign, method: str):
    diag = estimate_diagnostics(design)
    if method == "baseline_ols":
        return np.ones(design.X.shape[1]), 0., 1e-8, None
    if method == "lasso":
        return np.ones(design.X.shape[1]), .03, 0., None
    if method == "historical_proxy":
        raw, meta = historical_vclog_weights(design, diag)
        return np.clip(np.log1p(raw), .25, 4.), .03, 0., meta
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    weights = build_penalty_weights(diag, method, 1., (.25, 4.))
    return weights, .03, (.02 if method == "hybrid" else 0.), None


def _repo_revision() -> str:
    root = Path(__file__).resolve().parents[3]
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                   text=True).strip()


def evaluate_candidate_set(case_id: str, methods: Sequence[str], seed: int) -> list[dict[str, object]]:
    design, truth = make_fixture(case_id, seed)
    p = design.X.shape[1]
    masks = _candidate_masks(truth, p)
    digest = hashlib.sha256()
    for array in (design.X, design.y, design.sample_weight, design.environment_id.astype("U")):
        digest.update(np.ascontiguousarray(array).tobytes())
    dataset_sha = digest.hexdigest()
    code_sha = _repo_revision()
    split_path = Path(__file__).resolve().parents[1] / "configs" / "splits.yaml"
    split_sha = hash_file(split_path)
    rows: list[dict[str, object]] = []
    for method in methods:
        if method not in METHODS:
            raise ValueError(f"Unknown method {method}")
        beginning = time.perf_counter()
        weights, l1, l2, meta = _method_weights(design, method)
        fit = fit_weighted_en(design, l1=l1, l2=l2, weights=weights)
        criterion = {"baseline_ols": "residual", "lasso": "residual",
                     "historical_proxy": "historical_proxy",
                     "instability": "nir1_excess", "identifiability": "nir1_identifiability",
                     "hybrid": "nir1_identifiability"}[method]
        method_scores: dict[str, float] = {}
        for label, mask in masks.items():
            sub = ResearchDesign(design.X[:, mask], design.y, design.sample_weight,
                                 design.environment_id,
                                 tuple(t for t, use in zip(design.token_names, mask) if use),
                                 design.metadata)
            method_scores[label] = score_fixed_candidate(sub, criterion=criterion)
        ranked = sorted(method_scores, key=lambda name: (method_scores[name], name))
        truth_rank = (ranked.index("truth") + 1 if
                      np.isfinite(method_scores["truth"]) else None)
        elapsed = time.perf_counter() - beginning
        for label, mask in masks.items():
            config_sha = hashlib.sha256(json.dumps({"method": method, "l1": l1, "l2": l2},
                                                   sort_keys=True).encode()).hexdigest()
            run_spec = {"dataset": case_id, "variant": method, "noise": truth["noise_y"],
                        "data_seed": seed, "optimizer_seed": 0,
                        "config_sha": config_sha, "code_sha": code_sha,
                        "split_sha": split_sha, "candidate": label,
                        "dataset_sha": dataset_sha}
            finite = np.isfinite(method_scores[label])
            rows.append({
                "run_id": canonical_run_id(run_spec), "code_sha": code_sha,
                "config_sha": config_sha, "split_sha": split_sha,
                "dataset_sha": dataset_sha, "case": case_id, "seed": seed,
                "method": method, "candidate": label, "source_stage": "fixed_candidate",
                "candidate_criterion": criterion, "truth_model_class": truth["model_class"],
                "selected_by_search": False, "status": "ok",
                "candidate_score": float(method_scores[label]) if finite else None,
                "score_status": "scored" if finite else "unscorable",
                "truth_candidate_rank": truth_rank,
                "candidate_matches_truth": label == "truth",
                "selected_support": np.flatnonzero(fit.support).tolist(),
                "selected_support_exact": bool(np.array_equal(fit.support,
                                                               _candidate_masks(truth, p)["truth"])),
                "false_deleted": int(np.sum(_candidate_masks(truth, p)["truth"] & ~fit.support)),
                "false_included": int(np.sum(~_candidate_masks(truth, p)["truth"] & fit.support)),
                "solver_converged": fit.converged,
                "solver_kkt_residual": fit.kkt_residual,
                "diagnostic_assumptions": ("errors_in_variables" if truth["noise_X"] else
                                           "iid_response_noise_unverified"),
                "historical": meta, "elapsed_seconds": elapsed,
            })
    return rows
