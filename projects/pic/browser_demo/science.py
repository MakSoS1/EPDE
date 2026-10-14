"""Browser-only scientific engine: real PySINDy + an explicitly NON-EPDE toy evolution.

The original EPDE package is NOT imported, nor is this evolutionary sketch a
drop-in implementation of its token pool, MOEA/D, sparse operator or Pareto
logic. No symbolic-equation-success claim is made for it.
"""
from __future__ import annotations

import csv
import io
import json
import math
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter
from sklearn.preprocessing import PolynomialFeatures

CASES = {
    "decay": {"title": "Exponential decay", "variables": ["u"], "truth": ["u_t = -0.7 u"]},
    "logistic": {"title": "Logistic growth", "variables": ["u"], "truth": ["u_t = 1.2 u - 0.6 u²"]},
    "oscillator": {"title": "Damped harmonic oscillator", "variables": ["u", "v"], "truth": ["u_t = v", "v_t = -u - 0.2 v"]},
    "lotka": {"title": "Lotka–Volterra", "variables": ["u", "v"], "truth": ["u_t = 1.5 u - u v", "v_t = u v - 0.7 v"]},
    "vdp": {"title": "Van der Pol", "variables": ["u", "v"], "truth": ["u_t = v", "v_t = (1 - u²)v - u"]},
}
MAX_CSV_BYTES = 1_000_000
MAX_SAMPLES = 10_000
MAX_VARIABLES = 4


def _rhs(case):
    if case == "decay":
        return lambda t, y: [-0.7 * y[0]], [2.0], 7.0
    if case == "logistic":
        return lambda t, y: [1.2*y[0] - 0.6*y[0]**2], [0.25], 9.0
    if case == "oscillator":
        return lambda t, y: [y[1], -y[0] - 0.2*y[1]], [1.0, 0.0], 15.0
    if case == "lotka":
        return lambda t, y: [1.5*y[0] - y[0]*y[1], y[0]*y[1] - 0.7*y[1]], [1.5, 0.9], 12.0
    if case == "vdp":
        return lambda t, y: [y[1], (1-y[0]**2)*y[1]-y[0]], [2.0, 0.0], 16.0
    raise ValueError("Unknown built-in equation")


def _csv_data(text):
    if not isinstance(text, str) or len(text.encode("utf-8")) > MAX_CSV_BYTES:
        raise ValueError("CSV file is too large; maximum 1 MB")
    reader = csv.reader(io.StringIO(text, newline=""))
    rows = iter(reader)
    header = next(rows, None)
    if not header or not 2 <= len(header) <= MAX_VARIABLES+1:
        raise ValueError("CSV needs time in the first column and 1–4 variables")
    names = [str(x).strip() for x in header[1:]]
    if len(set(names)) != len(names) or any(
        not name or len(name) > 24 or not name.replace("_", "").isalnum()
        for name in names
    ):
        raise ValueError("CSV column names must be distinct simple identifiers")
    numbers = []
    for row in rows:
        if not row or all(not cell.strip() for cell in row):
            continue
        if len(row) != len(header):
            raise ValueError("Inconsistent CSV column count")
        numbers.append([float(cell) for cell in row])
        if len(numbers) > MAX_SAMPLES:
            raise ValueError("Too many CSV rows; limit 10,000")
    if len(numbers) < 80:
        raise ValueError("CSV must contain at least 80 samples")
    matrix = np.asarray(numbers, dtype=float)
    if not np.isfinite(matrix).all() or not np.all(np.diff(matrix[:, 0]) > 0):
        raise ValueError("Time must be strictly increasing and all values finite")
    return matrix[:, 0], matrix[:, 1:], names


def load_data(job):
    case = str(job.get("dataset", "decay"))
    if case == "csv":
        t, clean, names = _csv_data(job.get("csv", ""))
        truth = []
        title = "Your CSV"
    else:
        if case not in CASES:
            raise ValueError("Unsupported equation")
        fun, initial, stop = _rhs(case)
        count = min(MAX_SAMPLES, max(120, int(job.get("samples", 401))))
        t = np.linspace(0, stop, count)
        solve = solve_ivp(fun, (float(t[0]), float(t[-1])), initial, t_eval=t,
                          rtol=1e-10, atol=1e-12)
        if not solve.success:
            raise ValueError("Built-in ODE integration failed")
        clean = solve.y.T
        names = CASES[case]["variables"]
        truth = CASES[case]["truth"]
        title = CASES[case]["title"]
    noise = float(job.get("noise", 0.0))
    seed = int(job.get("seed", 0))
    if not 0 <= noise <= 20 or not 0 <= seed <= 100_000:
        raise ValueError("Noise must be 0–20%, seed 0–100,000")
    rng = np.random.default_rng(seed)
    observed = clean.copy()
    if noise:
        observed += rng.normal(size=observed.shape) * (noise / 100.0) * np.std(clean, axis=0)
    return t, observed, list(names), truth, title


def derivatives(t, X, method):
    if method == "finite":
        return np.gradient(X, t, axis=0, edge_order=2)
    if method == "savgol":
        widths = min(21, len(t) - (1 if len(t) % 2 == 0 else 0))
        widths = max(5, widths)
        if widths % 2 == 0:
            widths -= 1
        # Uses interpolation on uneven grids for the smoother, then gradient.
        if not np.allclose(np.diff(t), np.diff(t)[0], rtol=1e-4, atol=1e-9):
            from scipy.interpolate import interp1d
            regular_t = np.linspace(t[0], t[-1], len(t))
            smoothed = savgol_filter(interp1d(t, X, axis=0)(regular_t),
                                     window_length=widths, polyorder=3, axis=0)
            return np.gradient(smoothed, regular_t, axis=0)[
                np.searchsorted(regular_t, t).clip(0, len(t)-1)]
        dt = float(np.median(np.diff(t)))
        return savgol_filter(X, window_length=widths, polyorder=3, deriv=1,
                            delta=dt, axis=0)
    raise ValueError("Unsupported differentiation method")


def _scores(y, predicted):
    y = np.asarray(y, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    residual = y - predicted
    denom = float(np.sum((y - np.mean(y, axis=0))**2))
    r2 = 1.0 - float(np.sum(residual**2)) / (denom + 1e-20)
    return {"r2": r2, "rmse": float(np.sqrt(np.mean(residual**2)))}


def _format_equation(names, j, features, coefs):
    terms = []
    for name, c in zip(features, coefs):
        if abs(float(c)) > 1e-10:
            terms.append(f"{float(c):+.5g}·{name}")
    return f"d{names[j]}/dt = " + (" ".join(terms).lstrip("+") if terms else "0")


def _pysindy(X, dX, t, names, degree, threshold, cut):
    import pysindy as ps
    model = ps.SINDy(
        optimizer=ps.STLSQ(threshold=threshold, alpha=1e-6,
                            normalize_columns=True, max_iter=30),
        feature_library=ps.PolynomialLibrary(degree=degree, include_bias=True),
    )
    model.fit(X[:cut], t=t[:cut], x_dot=dX[:cut], feature_names=names)
    prediction = np.asarray(model.predict(X), dtype=float)
    features = list(model.get_feature_names())
    coefficients = np.asarray(model.coefficients(), dtype=float)
    if coefficients.shape != (X.shape[1], len(features)):
        raise ValueError("Unexpected PySINDy coefficient matrix shape")
    equations = [_format_equation(names, j, features, coefficients[j])
                 for j in range(X.shape[1])]
    return prediction, equations, int(np.count_nonzero(coefficients)), {
        "implementation": "PySINDy 2.1.0 / STLSQ",
        "features": features, "coefficients": coefficients.tolist(),
    }


def _evo_educational(X, dX, names, degree, cut, population, epochs, seed):
    """Small independently written evolutionary subset selector. NOT EPDE."""
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    features = poly.fit_transform(X)
    labels = list(poly.get_feature_names_out(names))
    if features.shape[1] > 55:
        raise ValueError("Library exceeds educational evolutionary search limit")
    train = features[:cut]
    valid = features[cut:]
    rng = np.random.default_rng(seed)
    all_coef, equations, history = [], [], []
    for target in range(X.shape[1]):
        yt, yv = dX[:cut, target], dX[cut:, target]
        memo = {}

        def evaluate(bits):
            key = bits.tobytes()
            if key in memo:
                return memo[key]
            active = np.flatnonzero(bits)
            if len(active) == 0:
                outcome = (float("inf"), None, 0)
            else:
                coef = np.linalg.lstsq(train[:, active], yt, rcond=1e-10)[0]
                yp = valid[:, active] @ coef
                error = float(np.mean((yp - yv)**2)) / (float(np.var(yv))+1e-10)
                outcome = (error + 0.0015*len(active), coef, len(active))
            memo[key] = outcome
            return outcome

        pool = []
        for _ in range(population):
            bits = rng.random(len(labels)) < min(0.5, 3.0/len(labels))
            if not np.any(bits):
                bits[int(rng.integers(len(bits)))] = True
            pool.append(bits)
        best = None
        for epoch in range(epochs):
            pool.sort(key=lambda b: evaluate(b)[0])
            if best is None or evaluate(pool[0])[0] < evaluate(best)[0]:
                best = pool[0].copy()
            history.append({"epoch": epoch+1, "variable": names[target],
                            "criterion": float(evaluate(best)[0]),
                            "active_terms": int(evaluate(best)[2])})
            elite = [p.copy() for p in pool[:max(2, population//4)]]
            next_pool = elite.copy()
            while len(next_pool) < population:
                mom, dad = rng.choice(len(elite), size=2, replace=True)
                mask = rng.random(len(labels)) < 0.5
                child = np.where(mask, elite[int(mom)], elite[int(dad)])
                flips = rng.random(len(labels)) < 0.12
                child = np.logical_xor(child, flips)
                if not np.any(child):
                    child[int(rng.integers(len(labels)))] = True
                next_pool.append(child)
            pool = next_pool
        active = np.flatnonzero(best)
        fit = np.linalg.lstsq(train[:, active], yt, rcond=1e-10)[0]
        coef = np.zeros(len(labels), dtype=float)
        coef[active] = fit
        all_coef.append(coef)
        equations.append(_format_equation(names, target, labels, coef))
    prediction = features @ np.asarray(all_coef, dtype=float).T
    return prediction, equations, int(sum(np.count_nonzero(c) for c in all_coef)), {
        "implementation": "EvoSparse educational subset selector — NOT EPDE",
        "features": labels,
        "coefficients": [c.tolist() for c in all_coef],
        "history": history,
    }


def run_job(raw):
    job = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(job, dict):
        raise ValueError("Expected JSON job object")
    started = time.perf_counter()
    method = job.get("method")
    if method not in ("pysindy", "evo_demo"):
        raise ValueError("This browser does not run native EPDE")
    degree = int(job.get("degree", 2))
    threshold = float(job.get("threshold", 0.05))
    population = int(job.get("population", 20))
    epochs = int(job.get("epochs", 8))
    if not 1 <= degree <= 4 or not 0 < threshold <= 2:
        raise ValueError("Degree or STLSQ threshold out of range")
    if not 4 <= population <= 80 or not 1 <= epochs <= 40:
        raise ValueError("Evolution resource limit exceeded")
    t, X, names, truth, title = load_data(job)
    dX = derivatives(t, X, str(job.get("derivatives", "finite")))
    cut = int(len(t)*0.8)
    if method == "pysindy":
        pred, equations, nonzeros, details = _pysindy(
            X, dX, t, names, degree, threshold, cut)
    else:
        pred, equations, nonzeros, details = _evo_educational(
            X, dX, names, degree, cut, population, epochs,
            int(job.get("seed", 0)))
    if not np.all(np.isfinite(pred)):
        raise ValueError("Non-finite model prediction")
    indices = np.unique(np.linspace(0, len(t)-1, min(len(t), 450)).astype(int))
    return json.dumps({
        "schema": "browser-demo/1.0",
        "status": "ok", "method": method, "title": title,
        "dataset": job.get("dataset"), "names": names, "truth": truth,
        "equations": equations, "active_terms": nonzeros,
        "train": _scores(dX[:cut], pred[:cut]),
        "validation": _scores(dX[cut:], pred[cut:]),
        "samples": len(t), "noise": float(job.get("noise", 0)),
        "seed": int(job.get("seed", 0)), "degree": degree,
        "derivatives": job.get("derivatives", "finite"),
        "threshold": threshold,
        "runtime_seconds": float(time.perf_counter() - started),
        "details": details,
        "data": {
            "t": np.asarray(t)[indices].tolist(),
            "observed": np.asarray(X)[indices].tolist(),
            "target_derivative": np.asarray(dX)[indices].tolist(),
            "model_derivative": np.asarray(pred)[indices].tolist(),
        },
        "scope": "In-sample derivatives and time-tail holdout; "
                 "NOT an out-of-trajectory prediction",
    }, allow_nan=False)
