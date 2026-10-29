"""Fast, deterministic starting advice; no fitting or reference-law selection.

Only already available geometry and explicitly declared added noise enter the
rules. The historical index describes execution experience, not tuned optima.
The existing scientific validators and hosted resource gate remain authoritative.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import json
from math import isfinite, log, prod
from pathlib import Path

import numpy as np

from support.discovery_lab.execution_policy import (
    MAX_CELLS, MAX_EPOCHS, MAX_POPULATION, MAX_SEARCH_WORK, MAX_VARIABLES,
)

VERSION = "starting-settings-v1"


def _uniform(coordinates):
    grid = np.asarray(coordinates)
    values = np.asarray(grid, dtype=float).reshape(-1)
    steps = np.diff(values)
    if not steps.size or not np.isfinite(values).all() or not (steps > 0).all():
        return False
    precision = np.finfo(grid.dtype).eps if grid.dtype.kind == "f" else np.finfo(float).eps
    if (precision < np.finfo(np.float32).eps
            and np.all(np.abs(values) <= np.finfo(np.float32).max)
            and np.array_equal(values, values.astype(np.float32).astype(float))):
        precision = np.finfo(np.float32).eps
    tolerance = min(4 * precision * float(np.max(np.abs(values))), abs(float(steps[0])) * 1e-3)
    return bool(np.allclose(steps, steps[0], rtol=1e-5, atol=tolerance))


def summarize_input(*, record_name=None, problem=None, arrays=None, noise_percent=0.0,
                    heavy=False):
    """Return a narrow JSON metadata object, never equations or field values.

    Heavy records use the registry's declared kind only. This path deliberately
    does not inspect arrays, load a dataset, or hash a scientific input file.
    """
    noise = float(noise_percent)
    if not isfinite(noise) or noise < 0:
        raise ValueError("Declared noise must be a finite nonnegative percentage")
    context = {"dataset": record_name, "kind": "unknown", "shape": [],
               "variable_count": None, "noise_percent": noise,
               "uniform_axes": [], "heavy": bool(heavy)}
    if heavy:
        from epde_bench.datasets import REGISTRY
        spec = REGISTRY.get(record_name)
        if spec:
            context["kind"] = spec.kind
        return context
    if problem is not None:
        context.update(kind=problem.kind, shape=[int(n) for n in problem.shape],
                       variable_count=len(problem.variables))
        axes = []
        for index, grid in enumerate(problem.grids):
            values = np.asarray(grid)
            if values.ndim > 1:
                slicing = [0] * values.ndim
                slicing[index] = slice(None)
                values = values[tuple(slicing)]
            axes.append(values)
    elif arrays is not None:
        axis_keys = sorted((k for k in arrays if k.startswith("axis_") and k != "axis_names"),
                           key=lambda k: int(k.removeprefix("axis_")))
        axes = [np.asarray(arrays[k]) for k in axis_keys]
        count = sum(k.startswith("var_") for k in arrays)
        context.update(shape=[int(a.size) for a in axes], variable_count=count,
                       kind={1: "ode" if count == 1 else "ode_system", 2: "pde_1d",
                             3: "pde_2d", 4: "pde_3d"}.get(len(axes), "unknown"))
    else:
        axes = []
    context["uniform_axes"] = [_uniform(a) for a in axes]
    return context


@lru_cache(maxsize=1)
def _evidence_index():
    try:
        value = json.loads(Path(__file__).with_name("starting_evidence.json").read_text())
        return value if value.get("version") == 1 else {}
    except (OSError, ValueError, AttributeError):
        return {}


def _historical_context(method, context):
    # The old PySINDy baseline is a different implementation from current
    # PDEComp. Its thresholds and outcomes must never migrate into its advice.
    if method != "epde" or not context.get("dataset"):
        return None
    entries = [e for e in _evidence_index().get("entries", [])
               if e.get("method") == "epde" and e.get("variant") == "default"
               and e.get("dataset") == context["dataset"]
               and e.get("noise_percent") == context.get("noise_percent", 0.0)]
    if not entries:
        return None
    return {"scope": "historical_clean_baseline", "used_for": "execution context only",
            "total": sum(e["total"] for e in entries),
            "completed": sum(e["completed"] for e in entries),
            "timed_out": sum(e["timed_out"] for e in entries),
            "source": _evidence_index().get("source", {})}


def _epde_budget(search, shape, hosted, reasons):
    evolution = search.setdefault("evolution", {})
    population = max(4, int(evolution.get("population_size", 16)))
    epochs = max(1, int(evolution.get("training_epochs", 5)))
    if not hosted:
        return
    cells = prod(shape) if shape else 0
    candidates = [
        (p, e) for p in sorted({4, 8, 12, 16, 20, 24, min(MAX_POPULATION, population)})
        for e in range(1, min(MAX_EPOCHS, epochs) + 1)
        if p <= population and (not cells or cells * p * e <= MAX_SEARCH_WORK)
    ]
    if not candidates:
        reasons.append("The input is too large for the hosted work limit even at a minimal budget; use a smaller input or local execution.")
        return
    # Project the saved budget onto feasible choices, preserving its scale.
    # This distance describes a resource adjustment, not a quality prediction.
    chosen = min(candidates, key=lambda pair: (
        log(pair[0] / population) ** 2 + log(pair[1] / epochs) ** 2,
        -(pair[0] * pair[1]), -pair[1]))
    evolution.update(population_size=chosen[0], training_epochs=chosen[1])
    if chosen != (population, epochs):
        reasons.append(f"Adjusted the saved budget from {population} × {epochs} to {chosen[0]} candidates × {chosen[1]} epochs for {cells:,} grid cells and the hosted work limit.")
    else:
        reasons.append(f"The profile's {population} candidates × {epochs} epochs fit this input's hosted work limit.")


def _epde_noise(search, shape, context, reasons):
    if float(context.get("noise_percent", 0)) < 1.0 or not shape:
        return
    prep = search.setdefault("preprocessing", {})
    if prep.get("default_preprocessor_type", "FD") != "FD":
        reasons.append("Kept the profile's chosen differentiation method; validate its noise sensitivity on this input.")
        return
    if context.get("uniform_axes") != [True] * len(shape):
        reasons.append("Kept finite differences because uniform spacing was not confirmed; check derivative quality before increasing the search budget.")
        return
    raw = prep.get("max_deriv_order", "auto")
    orders = ([2] + [4] * (len(shape) - 1) if raw == "auto"
              else [int(raw)] if isinstance(raw, int) else list(raw))
    degree = max(orders) + 1
    desired = 13 if float(context.get("noise_percent", 0)) >= 5 else 9
    available = min(33, min(shape) - 2)
    window = min(desired, available if available % 2 else available - 1)
    minimum = degree + 2
    minimum += int(minimum % 2 == 0)
    if window < minimum:
        window = minimum
    if window > available or window < 5:
        reasons.append("The grid is too short for a validated smoothing window at the requested derivative order; kept finite differences.")
        return
    from epde_bench.poly_options import polynomial_kwargs
    kwargs = polynomial_kwargs(window, degree, orders, shape)
    prep.update(default_preprocessor_type="poly", max_deriv_order=orders,
                preprocessor_kwargs=kwargs)
    reasons.append(f"For {float(context['noise_percent']):g}% declared added noise, suggested local polynomial differentiation (window {window}, degree {degree}). Check smoothing bias as well as noise reduction.")


def suggest_settings(method, base, context, *, protocol="native", hosted=True):
    """Suggest valid, editable starting values from the existing method profile.

    ``base`` is the editor's already cached full profile; this function never
    invokes PDEComp describe(), trains a model, or accesses reference metrics.
    """
    if method not in ("epde", "pdecomp", "deepmod", "discover"):
        raise ValueError("Unknown suggestion method")
    settings = deepcopy(base)
    shape = tuple(int(n) for n in context.get("shape", []))
    source = ("Saved equation profile" if context.get("dataset") else "General method defaults")
    if method != "epde":
        from epde_bench.pdecomp import PDECOMP_FILES
        source = "Pinned PDEComp " + ("dataset profile" if context.get("dataset") in PDECOMP_FILES
                                      else "method defaults")
    reasons = [source + "; these are starting values, not a measured optimum."]
    if context.get("heavy"):
        reasons.append("Metadata-only suggestion: kept the full profile for local execution; no large numerical dataset was loaded.")
    elif method == "epde":
        search = settings.setdefault("search", {})
        _epde_budget(search, shape, hosted, reasons)
        _epde_noise(search, shape, context, reasons)
        if (not shape or len(shape) > 2 or any(n < 10 for n in shape)
                or prod(shape) > MAX_CELLS or (context.get("variable_count") or 0) > MAX_VARIABLES):
            reasons.append("Budget suggestions cannot make unsupported input geometry eligible for hosted EPDE; the run-time resource gate still applies.")
    elif method == "pdecomp":
        field = "crop" if protocol == "fixed" else "boundary_trim"
        if shape and field in settings:
            limit = max(0, (min(shape) - 8) // 2)
            original = int(settings[field])
            settings[field] = min(original, limit)
            if settings[field] != original:
                reasons.append(f"Reduced boundary trimming from {original} to {settings[field]} points to retain enough interior samples on the shortest axis.")
        reasons.append("Kept the pinned sparse optimizer and library settings. The archive has no compatible parameter sweep for the current PySINDy implementation.")
    elif method == "deepmod":
        if protocol == "native" and hosted:
            settings["max_iterations"] = min(int(settings.get("max_iterations", 5000)), 250)
            cells = prod(shape) if shape else 1000
            settings["max_samples"] = max(100, min(int(settings.get("max_samples", 10000)), 1000, cells))
            reasons.append(f"Start with {settings['max_iterations']} training iterations and at most {settings['max_samples']} training samples for a short hosted run; the runner also caps sampling to the available data.")
            reasons.append("Short training is a first diagnostic. Increase iterations and validate the discovered equation before drawing a quality conclusion.")
        elif protocol == "fixed":
            reasons.append("Kept the fixed-protocol sparse estimator settings; neural training controls do not apply to this protocol.")
        else:
            reasons.append("Kept the full native training profile for local execution.")
    elif method == "discover":
        if hosted:
            settings["n_samples"] = min(int(settings.get("n_samples", 5000)), 1000)
            settings["batch_size"] = min(int(settings.get("batch_size", 250)), 100, settings["n_samples"])
            reasons.append(f"Start with {settings['n_samples']} expression samples in batches of {settings['batch_size']}; keep the profile's symbolic operators and risk-seeking fraction.")
            reasons.append("A small search budget can miss good expressions. Compare several seeds and a larger budget after checking the first result.")
    if method == "discover" or method == "deepmod" and protocol == "fixed":
        if False in context.get("uniform_axes", []):
            reasons.append("This protocol requires uniformly spaced coordinates; resample the input before running it.")
    evidence = _historical_context(method, context)
    if evidence:
        reasons.append(f"Historical clean-data baseline for this record: {evidence['completed']}/{evidence['total']} completed, {evidence['timed_out']} timed out. It used an older source/environment and one setting across seeds; it does not establish the best parameters or predict this server's speed.")
    elif method != "epde":
        reasons.append("No compatible archived tuning runs are available for this method; the suggestion uses its profile and input/resource rules.")
    return {"version": VERSION, "method": method, "protocol": protocol,
            "settings": settings, "reasons": reasons, "source": source,
            "evidence": evidence}
