"""T1–T8 preregistered synthetic controls (never masquerade as PIC measurements)."""

from __future__ import annotations

import numpy as np

from .design import ResearchDesign


def make_fixture(case_id: str, seed: int = 0, *, correlation: float = 0.99) -> tuple[ResearchDesign, dict[str, object]]:
    """Return measured design and ground truth separately to prevent oracle leakage.

    Noise signal, derivative perturbation and environment distribution each
    receive an independent random stream. The ground-truth dictionary must
    never be passed to the model-selection implementation.
    """
    if case_id not in {f"T{i}" for i in range(1, 9)}:
        raise ValueError(f"Unknown fixture: {case_id}")
    if not 0 <= correlation < 1:
        raise ValueError("correlation must be in [0, 1)")
    rng_x, rng_y, rng_deriv = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3))
    n, p = 512, 4
    env = np.repeat(np.arange(4).astype(str), n // 4)
    X = rng_x.standard_normal((n, p))
    beta = np.array([1.5, -0.7, 0., 0.])
    noise_y, noise_x = .03, 0.
    model_class = "constant"
    notes: dict[str, object] = {}
    if case_id == "T1":
        Q, _ = np.linalg.qr(X)
        X = np.sqrt(n) * Q  # exactly orthogonal columns
    elif case_id == "T2":
        X[:, 1] = correlation * X[:, 0] + np.sqrt(1 - correlation**2) * X[:, 1]
        beta = np.array([1.5, 0., .3, 0.])
        notes["correlation"] = float(correlation)
    elif case_id == "T3":
        X[:, 1] *= 10000.
        beta = np.array([1.5, 1e-4, 0., 0.])
        notes["feature_scales"] = [1., 10000., 1., 1.]
    elif case_id == "T4":
        X[:, 2] = .96 * X[:, 0] + np.sqrt(1 - .96**2) * X[:, 2]
        notes["candidate_supports"] = [[0, 1], [1, 2], [0, 1, 2]]
    elif case_id == "T5":
        for k in range(4):
            mask = env == str(k)
            X[mask] = (1. + .35 * k) * X[mask] + .25 * k
        notes["independent_initial_conditions"] = 4
    elif case_id == "T6":
        model_class = "variable"
        beta = np.array([1.5, 0., 0., 0.])
        notes["coefficient_by_environment"] = {str(k): 1.5 + .6 * k for k in range(4)}
    elif case_id == "T7":
        noise_y, noise_x = .1, .15
        notes["errors_in_variables"] = True
    elif case_id == "T8":
        X[:, 1] = X[:, 0]
        beta = np.array([1.5, 0., .3, 0.])
        notes["exact_duplicate"] = [0, 1]

    if case_id == "T6":
        y = np.array([notes["coefficient_by_environment"][group] * X[i, 0]
                      for i, group in enumerate(env)])
    else:
        y = X @ beta
    y = y + noise_y * rng_y.standard_normal(n)
    X_observed = X + noise_x * rng_deriv.standard_normal(X.shape) if noise_x else X
    design = ResearchDesign(X_observed, y, np.ones(n), env,
                            tuple(f"term_{i}" for i in range(p)),
                            {"fixture": case_id, "seed": seed, "noise_y": noise_y,
                             "noise_X": noise_x, "model_class": model_class})
    truth: dict[str, object] = {"beta": beta.tolist(),
                                "support": np.flatnonzero(beta != 0).tolist(),
                                "model_class": model_class, "noise_y": noise_y,
                                "noise_X": noise_x, **notes}
    return design, truth
