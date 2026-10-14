"""H8 falsification test: would *one* multi-mode trajectory already suffice?

S1's analytic PDE trajectories contained two Fourier modes; the positive
fixed-target H8 study instead made each of its single-IC trajectories a
SINGLE Fourier mode. u_xx = -k^2 u exactly on such an input, making the
fixed dictionary unidentifiable. The multi arm combines different k.

This test keeps the estimator, 0.0005 complexity penalty, derivative
order, feature library, and approximate training sample budget unchanged,
but gives EACH trajectory two nonzero spatial harmonics from the start.

All ten tasks in S2e and their full evolutionary EPDE are distinct from
this fixed-target mechanical probe. The 5 random seeds are correlated
repeats, not five independent physical systems. No algorithm hyperparameter
may be tuned to this test before presenting its result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .multitrajectory_identifiability import (
    TRUTH, choose_support, fixed_dictionary,
)

PDE_CASES = ("heat", "reactiondiff", "advdiff")


def two_mode_trajectories(case: str, seed: int):
    if case not in PDE_CASES:
        raise ValueError("Only predeclared PDE systems are supported")
    rng = np.random.default_rng(seed)
    t = np.linspace(0., 5., 241)
    x = np.linspace(-np.pi, np.pi, 112, endpoint=False)
    tt, xx = np.meshgrid(t, x, indexing="ij")
    result = []
    for j in range(4):
        u = np.zeros_like(tt)
        for index, k in enumerate((j + 1, j + 2)):
            amplitude = (1. if index == 0 else .38) * (1. + rng.uniform(-.04, .04))
            phase = rng.uniform(-.2, .2)
            growth = {
                "heat": -.17 * k**2,
                "reactiondiff": .35 - .19 * k**2,
                "advdiff": -.13 * k**2,
            }[case]
            drift = .65 if case == "advdiff" else 0.
            u += amplitude * np.exp(growth * tt) * np.sin(
                k * (xx - drift * tt) + phase)
        result.append((t, x, u))
    return result


def one_case(case, seed):
    parts = [fixed_dictionary(x, target_time_order=1)
             for x in two_mode_trajectories(case, seed)]
    single_t = [(parts[0][0], parts[0][1][::3], parts[0][2][::3])]
    single_v = [(parts[0][0], parts[0][1][1::3], parts[0][2][1::3])]
    multi_t = [(row[0], row[1][::9], row[2][::9]) for row in parts[:3]]
    multi_v = [(row[0], row[1][1::9], row[2][1::9]) for row in parts[:3]]
    a = choose_support(single_t, single_v)
    b = choose_support(multi_t, multi_v)

    def on_unseen_initial(model):
        # A whole fourth trajectory is held out until both models have
        # already selected their supports and coefficients.
        _, X, y = parts[3]
        prediction = X[:, model["indices"]] @ model["coefficients"]
        return float(np.mean((prediction - y)**2) / (np.var(y) + 1e-12))

    truth = set(TRUTH[case])
    def verdict(model):
        return {
            "support": list(model["support"]),
            "exact": bool(set(model["support"]) == truth),
            "heldout_relative_mse": on_unseen_initial(model),
            "training_rows": int(model["train_samples"]),
            "validation_rows": int(model["validation_samples"]),
        }
    if abs(a["train_samples"] - b["train_samples"]) > 3:
        raise AssertionError("Unequal observation budget outside predeclared tolerance")
    return {"case": case, "seed": seed,
            "single": verdict(a), "multi": verdict(b),
            "truth_used_for_selection": False,
            "same_predeclared_library": True,
            "two_spatial_modes_in_EACH_initial_condition": True}


def spatial_identity_test():
    """Exact derivatives demonstrate why EPDE preferred a spurious equation.

    The spatial 'law' u_x*u_xx = u*u_xxx is identically true for each
    individual sinusoid, independently of its temporal dynamics. A mix of
    two different harmonics violates it, providing a falsification probe.
    """
    x = np.linspace(-np.pi, np.pi, 4096, endpoint=False)

    def residual(modes):
        u = sum(a*np.sin(k*x) for k,a in modes)
        ux = sum(a*k*np.cos(k*x) for k,a in modes)
        uxx = sum(-a*k*k*np.sin(k*x) for k,a in modes)
        uxxx = sum(-a*k**3*np.cos(k*x) for k,a in modes)
        left, right = ux*uxx, u*uxxx
        rms = np.sqrt(np.mean((left-right)**2))
        denom = (np.sqrt(np.mean(left**2)) +
                 np.sqrt(np.mean(right**2))) / 2
        return {"absolute_rms": float(rms),
                "normalized_rms": float(rms / max(denom, 1e-12))}
    return {"one_mode": residual(((1,1.),)),
            "mixed_modes": residual(((1,1.), (2,.38)))}


def experiment(seeds=range(5)):
    rows = [one_case(case, seed) for case in PDE_CASES for seed in seeds]
    singles = sum(int(r["single"]["exact"]) for r in rows)
    multis = sum(int(r["multi"]["exact"]) for r in rows)
    return {
        "status": "MECHANISTIC_FALSIFICATION_ON_VIEWED_H8_FAMILIES_NOT_NEW_CONFIRMATION",
        "n_physical_families": len(PDE_CASES),
        "n_seed_repeats": len(rows),
        "all_true_single": singles, "all_true_multi": multis,
        "difference_pp": 100 * (multis - singles) / len(rows),
        "spurious_spatial_identity": spatial_identity_test(),
        "rows": rows,
        "limitations": [
            "Known time-derivative order and restricted fixed feature dictionary.",
            "Both arms use approximate same observed points, but validation "
            "neighbors share derivative stencils with training points.",
            "Entire fourth initial condition held out; not an independent "
            "physical law.",
            "This challenge was designed after examining H8 errors, "
            "so it is mechanistic diagnosis, not untouched S3 confirmation.",
            "Does not demonstrate an improvement of unrestricted EPDE."],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.seeds <= 20:
        parser.error("seeds must be between 1 and 20")
    result = experiment(range(args.seeds))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({k: result[k] for k in (
        "status", "n_seed_repeats", "all_true_single", "all_true_multi",
        "difference_pp")}))


if __name__ == "__main__":
    main()
