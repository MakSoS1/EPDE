#!/usr/bin/env python3
"""Bounded, reproducible multisample experiments using EPDE's real fit paths.

Fixed-support, assignment selection, and evolutionary discovery are separate
stages. Analytic fixture checks are explicitly *not* EPDE discovery results.
Every run has an isolated process, a timeout, a log, and a JSON outcome.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import traceback

THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")
for _key in THREAD_KEYS:
    os.environ[_key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("MPLBACKEND", "Agg")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np

SCHEMA_VERSION = 1
SUPPORT_TOL = 1e-8
SCENARIOS = ("affine", "logistic", "coupled", "reaction_diffusion",
             "contradictory", "density")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def finite_json(value):
    """Strict JSON; retain the locations/reasons of all unavailable numbers."""
    unavailable = []

    def convert(item, path):
        if isinstance(item, dict):
            return {str(k): convert(v, f"{path}.{k}") for k, v in item.items()}
        if isinstance(item, (list, tuple, np.ndarray)):
            return [convert(v, f"{path}[{i}]") for i, v in enumerate(item)]
        if isinstance(item, (float, np.floating)):
            if not math.isfinite(float(item)):
                unavailable.append({"path": path, "reason": "nonfinite numerical value"})
                return None
            return float(item)
        if isinstance(item, (bool, np.bool_)):
            return bool(item)
        if isinstance(item, (int, np.integer)):
            return int(item)
        if isinstance(item, Path):
            return str(item)
        if callable(item):
            return f"{item.__module__}.{item.__qualname__}"
        return item

    result = convert(value, "$")
    if unavailable and isinstance(result, dict):
        result.setdefault("unavailable_values", []).extend(unavailable)
    return result


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(finite_json(value), indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def derivative_names(var, orders):
    return [f"d{var}/dx{axis}" if order == 1 else f"d^{order}{var}/dx{axis}^{order}"
            for axis, maximum in enumerate(orders) for order in range(1, maximum + 1)]


def sample(sample_id, axes, fields, derivatives, orders, noise, rng, parameters):
    grids = list(np.meshgrid(*axes, indexing="ij")) if len(axes) > 1 else [axes[0]]
    tokens = {name: np.array(values, copy=True) for name, values in fields.items()}
    analytic, observed, noise_scales = {}, {}, {}
    for name, values in fields.items():
        names = derivative_names(name, orders)
        analytic[name] = np.column_stack([np.asarray(derivatives[name][label]).ravel()
                                          for label in names])
        tokens.update(derivatives[name])
        sigma = noise * float(np.std(values))
        observed[name] = values + rng.normal(0.0, sigma, values.shape) if sigma else values.copy()
        noise_scales[name] = sigma
    return {"id": sample_id, "axes": axes, "grids": grids, "fields": observed,
            "clean_fields": fields, "analytic_derivatives": analytic,
            "clean_tokens": tokens, "parameters": parameters, "noise_std": noise_scales}


def make_case(spec):
    """Actual analytic solutions, with independent grids and held-out records."""
    name, noise = spec["scenario"], spec["noise"]
    rng = np.random.default_rng(spec["seed"])
    orders = (1, 2) if name == "reaction_diffusion" else ((2,) if name == "density" else (1,))
    variables = ["u", "v"] if name == "coupled" else ["u"]
    targets = {var: f"d{var}/dx0" for var in variables}
    truth = {"affine": {"u": {"1": 0.4, "u": -0.7}},
             "logistic": {"u": {"u": 1.2, "u^2": -0.6}},
             "coupled": {"u": {"u": -0.2, "v": 1.0}, "v": {"u": -1.0, "v": -0.2}},
             "reaction_diffusion": {"u": {"u": -0.2, "d^2u/dx1^2": 0.1}},
             "contradictory": None, "density": None}[name]
    train, holdout = [], []
    if name == "reaction_diffusion":
        settings = [(7, 31, 41, 0.0, -np.pi, [(1, 1.0)]),
                    (19, 23, 53, 10.0, 4 * np.pi, [(2, 0.7)]),
                    (97, 29, 47, 23.0, 8 * np.pi + 0.17, [(1, 0.8), (3, 0.4)])]
        for ident, nt, nx, start, left, modes in settings:
            t, x = np.linspace(start, start + 2, nt), np.linspace(left, left + 2 * np.pi, nx)
            T, X = np.meshgrid(t - start, x, indexing="ij")
            u, dt, dx, dxx = (np.zeros_like(T) for _ in range(4))
            for k, amplitude in modes:
                envelope = amplitude * np.exp((-0.2 - 0.1 * k * k) * T)
                mode = envelope * np.sin(k * X)
                u += mode
                dt += (-0.2 - 0.1 * k * k) * mode
                dx += k * envelope * np.cos(k * X)
                dxx -= k * k * mode
            record = sample(ident, [t, x], {"u": u},
                            {"u": {"du/dx0": dt, "du/dx1": dx, "d^2u/dx1^2": dxx}},
                            orders, noise if ident != 97 else 0.0, rng,
                            {"modes": modes, "rho": -0.2, "nu": 0.1})
            (holdout if ident == 97 else train).append(record)
    elif name == "density":
        targets = {"u": "d^2u/dx0^2"}
        second_n = 256 if spec.get("density") == "refined" else 64
        for ident, n, omega in [(7, 64, 1.0), (19, second_n, 2.0), (97, 80, 1.5)]:
            t = np.linspace(0, 2 * np.pi / omega, n, endpoint=False)
            u = np.sin(omega * t)
            record = sample(ident, [t], {"u": u},
                            {"u": {"du/dx0": omega * np.cos(omega * t),
                                   "d^2u/dx0^2": -omega * omega * u}},
                            orders, 0.0, rng, {"omega": omega, "coefficient": -omega * omega})
            (holdout if ident == 97 else train).append(record)
    else:
        settings = [(7, 81, 0.0, 4.0, 0), (19, 151, 10.0, 5.0, 1),
                    (31, 301, 20.0, 3.0, 2), (97, 137, 40.0, 4.5, 3)]
        if name == "coupled":
            settings = [(7, 101, 0.0, 6.0, 0), (19, 173, 10.0, 7.0, 1),
                        (97, 137, 30.0, 6.5, 2)]
        if name == "contradictory":
            settings = [(7, 61, 0.0, 3.0, 0), (19, 181, 10.0, 3.0, 1),
                        (97, 101, 30.0, 3.0, 2)]
        for ident, n, start, duration, index in settings:
            t = np.linspace(start, start + duration, n)
            tau = t - start
            if name == "affine":
                initial = [0.0, 1.2, 2.5, 1.8][index]
                u = 4 / 7 + (initial - 4 / 7) * np.exp(-0.7 * tau)
                fields, deriv = {"u": u}, {"u": {"du/dx0": 0.4 - 0.7 * u}}
                params = {"initial": initial, "constant": 0.4, "linear": -0.7}
            elif name == "logistic":
                initial = [0.1, 0.6, 2.8, 1.3][index]
                u = 2 / (1 + (2 / initial - 1) * np.exp(-1.2 * tau))
                fields, deriv = {"u": u}, {"u": {"du/dx0": 1.2 * u - 0.6 * u * u}}
                params = {"initial": initial, "growth": 1.2, "capacity": 2.0}
            elif name == "coupled":
                u0, v0 = [(1.0, 0.0), (0.0, 2.0), (1.3, -0.8)][index]
                u = np.exp(-0.2 * tau) * (u0 * np.cos(tau) + v0 * np.sin(tau))
                v = np.exp(-0.2 * tau) * (v0 * np.cos(tau) - u0 * np.sin(tau))
                fields = {"u": u, "v": v}
                deriv = {"u": {"du/dx0": -0.2 * u + v},
                         "v": {"dv/dx0": -u - 0.2 * v}}
                params = {"initial": [u0, v0], "decay": 0.2, "frequency": 1.0}
            else:
                rate = [0.4, 1.0, 0.7][index]
                u = np.exp(-rate * tau)
                fields, deriv = {"u": u}, {"u": {"du/dx0": -rate * u}}
                params = {"initial": 1.0, "rate": rate}
            record = sample(ident, [t], fields, deriv, orders,
                            noise if ident != 97 else 0.0, rng, params)
            (holdout if ident == 97 else train).append(record)
    return {"name": name, "variables": variables, "orders": orders, "targets": targets,
            "truth": truth, "train": train, "holdout": holdout,
            "boundary": 0 if name == "density" else 2}


def describe_sample(record):
    return {"id": record["id"], "shape": list(next(iter(record["fields"].values())).shape),
            "grid_ranges": [[float(x[0]), float(x[-1]), len(x)] for x in record["axes"]],
            "parameters": record["parameters"], "noise_std": record["noise_std"]}


def term_key(factors):
    return " * ".join(label if power == 1 else f"{label}^{power:g}"
                      for label, power in sorted(factors.items())) or "1"


def term_record(term):
    factors = {}
    for factor in term.structure:
        power = next(float(factor.params[i]) for i, desc in factor.params_description.items()
                     if desc["name"] == "power")
        factors[factor.label] = factors.get(factor.label, 0.0) + power
    return {"key": term_key(factors), "factors": factors}


def evaluate_terms(terms, tokens):
    shape = next(iter(tokens.values())).shape
    total = np.zeros(shape)
    for term in terms:
        values = np.ones(shape)
        for label, power in term["factors"].items():
            values = values * np.asarray(tokens[label]) ** power
        total += term["coefficient"] * values
    return total


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else None


def matrix_diagnostics(matrix):
    singular = np.linalg.svd(matrix, compute_uv=False)
    rank = int(np.linalg.matrix_rank(matrix))
    return {"rows": matrix.shape[0], "columns": matrix.shape[1], "rank": rank,
            "singular_values": singular.tolist(),
            "condition_number": float(np.linalg.cond(matrix)) if rank == matrix.shape[1] else None,
            "condition_reason": None if rank == matrix.shape[1] else "rank deficient"}


def fixture_evidence(case):
    """Analytic identities/rank proof, labelled independently of EPDE fits."""
    evidence = {"kind": "analytic_fixture_check_not_EPDE_fit"}
    if case["name"] == "reaction_diffusion":
        matrices = [np.column_stack([s["clean_tokens"]["u"].ravel(),
                                    s["clean_tokens"]["d^2u/dx1^2"].ravel()])
                    for s in case["train"]]
        evidence["individual_designs"] = [matrix_diagnostics(m) for m in matrices]
        evidence["pooled_design"] = matrix_diagnostics(np.vstack(matrices))
        evidence["identity"] = "u_xx=-k^2*u for each single mode; distinct k remove the common nullspace"
    if case["name"] in ("contradictory", "density"):
        evidence["has_common_exact_coefficient"] = False
        evidence["individual_true_coefficients"] = [
            s["parameters"]["coefficient"] if case["name"] == "density" else -s["parameters"]["rate"]
            for s in case["train"]]
        if case["name"] == "density":
            evidence["sample_mean_squares"] = [float(np.mean(s["clean_fields"]["u"] ** 2))
                                                for s in case["train"]]
    if case["truth"] is not None:
        errors = []
        for record in case["train"] + case["holdout"]:
            for var, coefficients in case["truth"].items():
                rhs = np.zeros_like(record["clean_fields"][var])
                for key, value in coefficients.items():
                    values = (1 if key == "1" else record["clean_tokens"]["u"] ** 2
                              if key == "u^2" else record["clean_tokens"][key])
                    rhs = rhs + value * values
                errors.append(float(np.max(np.abs(rhs - record["clean_tokens"][case["targets"][var]]))))
        evidence["maximum_exact_law_residual"] = max(errors)
    return evidence


def make_search(case, spec):
    import epde
    config = {
        "objectives": {"multisample_mode": spec["multisample_mode"],
                       "multisample_weighting": spec["weighting"],
                       "discrepancy_metric": "l2_relative", "instability_metric": "dyad"},
        "domain": {"evaluation_form": spec["evaluation_form"]},
        "runtime": {"free_tensor_cache_after_fit": False},
    }
    search = epde.EpdeSearch(
        config=config, device="cpu", use_solver=False, engine="nsga2",
        system_rps_mode=spec["rps_mode"],
        system_rps_params={"plain_target": True, "target_axes": 0}
        if spec["rps_mode"] == "plain" else {},
        default_preprocessor_type="FD", boundary_width=case["boundary"],
        equation_terms_max_number=4, equation_factors_max_number=1,
        data_fun_pow=2 if case["name"] == "logistic" else 1, deriv_fun_pow=1,
        population_size=spec["population"], training_epochs=spec["epochs"],
        early_stopping_patience=None, verbose_params={"show_iter_idx": False})
    trajectories = []
    for record in case["train"]:
        domain = search.createDomain(record["grids"], ID=record["id"],
                                     gfunction=lambda grids: np.ones_like(grids[0]))[1]
        derivatives = record["analytic_derivatives"] if spec["derivatives"] == "analytic" else None
        trajectories.append(search.createTrajectory(record["fields"], domain,
                            cache_id=record["id"], derivs=derivatives)[1])
    search.create_pool(trajectories, max_deriv_order=case["orders"],
                       data_fun_pow=2 if case["name"] == "logistic" else 1, deriv_fun_pow=1)
    if spec["stage"] != "discovery":
        search._prepare_weak_rows()
    return search, trajectories


def known_candidate(case, pool):
    """Supply structure, with arbitrary initial coefficients that are reset."""
    from epde.interface.equation_translator import translate_equation
    features = {"affine": {"u": ["u"]}, "logistic": {"u": ["u", "u^2"]},
                "coupled": {"u": ["u", "v"], "v": ["u", "v"]},
                "reaction_diffusion": {"u": ["u", "d^2u/dx1^2"]},
                "contradictory": {"u": ["u"]}, "density": {"u": ["u"]}}[case["name"]]
    texts = {}
    for var, keys in features.items():
        tokens = ["u{power: 2.0}" if key == "u^2" else key + "{power: 1.0}" for key in keys]
        texts[var] = " + ".join("1.0 * " + token for token in tokens)
        texts[var] += " + 0.0 = " + case["targets"][var] + "{power: 1.0}"
    system = translate_equation(texts, pool, all_vars=case["variables"])
    for eq in system.vals:
        target = eq.target
        eq.reset_state(True)
        eq.target_idx = eq.structure.index(target)
    return system


def selector_components():
    from epde.interface.search_config import active_config
    from epde.operators.common.coeff_calculation import LinRegBasedCoeffsEquation
    from epde.operators.common.fitness import SolverFreeFitness
    from epde.operators.common.objectives import Discrepancy, Instability
    from epde.operators.common.right_part_selection import EqRightPartSelector, SoEqRightPartSelector
    from epde.operators.common.sparsity import build_sparsity_operator
    from epde.operators.utils.template import add_base_param_to_operator
    disc = Discrepancy()
    host = SolverFreeFitness(["penalty_coeff"], objectives=[disc, Instability()], primary=disc)
    add_base_param_to_operator(host, {})
    config = active_config().objectives
    coefficient_fit = LinRegBasedCoeffsEquation()
    equation_selector = EqRightPartSelector()
    equation_selector.set_suboperators({"fitness_calculation": host,
        "sparsity": build_sparsity_operator(config.sparsity_cls, config.sparsity_kwargs),
        "coeff_calc": coefficient_fit})
    selector = SoEqRightPartSelector()
    selector.set_suboperators({"eq_right_part_selector": equation_selector})
    return selector, coefficient_fit, host


def fit_fixed(case, search, spec):
    from epde.operators.common.plain_fit import _versions
    system = known_candidate(case, search.pool)
    selector, coefficient_fit, host = selector_components()
    diagnostics, fitted = {}, []
    for eq in system.vals:
        target, features = eq.evaluate()
        diagnostics[eq.main_var_to_explain] = {
            "individual_designs": {str(k): matrix_diagnostics(v) for k, v in features.items()},
            "pooled_design": matrix_diagnostics(np.vstack(list(features.values())))}
    if spec["stage"] == "assignment":
        for eq in system.vals:
            eq.reset_state(True)
        selector.apply(system, {})
        fitted = list(system.vals)
    else:
        for eq in system.vals:
            cap = len(eq.structure) if case["name"] == "affine" else len(eq.structure) - 1
            versions = _versions(eq, cap, coefficient_fit, {}, host, {})
            if not versions:
                return [], diagnostics
            fitted.append(min(versions, key=lambda variant: float(variant.fitness_value)))
    return [fitted], diagnostics


def equation_report(eq, case):
    var = eq.main_var_to_explain
    terms = []
    for i, term in enumerate(eq.structure):
        record = term_record(term)
        record["coefficient"] = (-1.0 if i == eq.target_idx
                                  else float(eq.weights_final[eq.weight_index(i)]))
        terms.append(record)
    terms.append({"key": "1", "factors": {}, "coefficient": float(eq.weights_final[-1])})
    combined = {}
    for term in terms:
        combined[term["key"]] = combined.get(term["key"], 0.0) + term["coefficient"]
    target_label = case["targets"][var]
    target_coefficient = combined.get(target_label, 0.0)
    canonical = None
    if abs(target_coefficient) > SUPPORT_TOL:
        canonical = {k: -v / target_coefficient for k, v in combined.items() if k != target_label}
    result = {"variable": var, "text": eq.text_form, "terms_in_zero_form": terms,
              "target": eq.target.name, "canonical_target": target_label,
              "canonical_coefficients": canonical,
              "canonical_reason": None if canonical is not None else "no linear canonical target term",
              "fitness": float(eq.fitness_value), "instability": float(eq.coefficients_stability),
              "coefficient_error": None, "support_exact": None,
              "coefficient_reason": "no common ground-truth coefficient vector" if case["truth"] is None else None}
    if case["truth"] is not None and canonical is not None:
        truth = case["truth"][var]
        keys = sorted(set(truth) | set(canonical))
        actual, expected = np.array([canonical.get(k, 0.0) for k in keys]), np.array([truth.get(k, 0.0) for k in keys])
        result["coefficient_error"] = ratio(np.linalg.norm(actual - expected), np.linalg.norm(expected))
        threshold = SUPPORT_TOL * max(1.0, max(abs(v) for v in canonical.values()))
        result["support_exact"] = {k for k, v in canonical.items() if abs(v) > threshold} == set(truth)
    elif case["truth"] is not None:
        result["coefficient_reason"] = "no linear canonical target term"
    targets, features = eq.evaluate()
    result["training_fit_residuals"] = []
    for ident, y in targets.items():
        resid = features[ident] @ np.asarray(eq.weights_final[:-1]) + eq.weights_final[-1] - y
        result["training_fit_residuals"].append({"sample_id": ident, "rows": len(y),
            "rmse": float(np.sqrt(np.mean(resid * resid))),
            "relative_l2": ratio(np.linalg.norm(resid), np.linalg.norm(y))})
    result["clean_training_residuals"], result["holdout_residuals"] = [], []
    for group, records in [("clean_training_residuals", case["train"]), ("holdout_residuals", case["holdout"])]:
        for record in records:
            residual = evaluate_terms(terms, record["clean_tokens"])
            target = record["clean_tokens"][target_label]
            canonical_residual = residual / abs(target_coefficient) if canonical is not None else residual
            result[group].append({"sample_id": record["id"],
                "relative_l2": ratio(np.linalg.norm(canonical_residual), np.linalg.norm(target)),
                "rmse": float(np.sqrt(np.mean(canonical_residual ** 2))),
                "normalization": "canonical target" if canonical is not None else "implicit equation scale"})
    return result


def rollout_report(equations, case):
    if len(case["orders"]) != 1 or case["orders"] != (1,):
        return {"relative_l2": None, "reason": "rollout implemented only for first-order ODE systems"}
    variables = case["variables"]
    parsed = {}
    for equation in equations:
        coefficients = equation["canonical_coefficients"]
        if coefficients is None:
            return {"relative_l2": None, "reason": "candidate is not explicit in its first time derivative"}
        terms = []
        for key, coefficient in coefficients.items():
            if abs(coefficient) <= SUPPORT_TOL:
                continue
            factors = next(term["factors"] for term in equation["terms_in_zero_form"] if term["key"] == key)
            if any(label not in variables for label in factors):
                return {"relative_l2": None, "reason": "RHS contains derivative features"}
            terms.append((coefficient, factors))
        parsed[equation["variable"]] = terms
    from scipy.integrate import solve_ivp
    record = case["holdout"][0]
    times = record["axes"][0]
    initial = [record["clean_fields"][v][0] for v in variables]

    def rhs(_, state):
        values = dict(zip(variables, state))
        return [sum(c * math.prod(values[label] ** power for label, power in factors.items())
                    for c, factors in parsed[var]) for var in variables]

    try:
        solution = solve_ivp(rhs, (times[0], times[-1]), initial, t_eval=times, rtol=1e-8, atol=1e-10)
        if not solution.success or solution.y.shape[1] != len(times):
            return {"relative_l2": None, "reason": "integration failed", "message": solution.message}
        truth = np.vstack([record["clean_fields"][v] for v in variables])
        return {"relative_l2": ratio(np.linalg.norm(solution.y - truth), np.linalg.norm(truth)), "reason": None}
    except Exception as exc:
        return {"relative_l2": None, "reason": "integration raised", "message": str(exc)}


def run_worker(spec):
    start = time.perf_counter()
    result = {"spec": spec, "status": "error", "started_at": utc_now(), "models": []}
    try:
        random.seed(spec["seed"])
        np.random.seed(spec["seed"])
        import torch
        torch.manual_seed(spec["seed"])
        torch.set_num_threads(1)
        case = make_case(spec)
        result.update(train_samples=[describe_sample(s) for s in case["train"]],
                      heldout_samples=[describe_sample(s) for s in case["holdout"]],
                      fixture_evidence=fixture_evidence(case))
        search, trajectories = make_search(case, spec)
        result["resolved_config"] = search._config.as_dict()
        if spec["stage"] == "discovery":
            search.fit(data=trajectories, max_deriv_order=case["orders"])
            systems = list(search.optimizer.reported_front())
            models = [list(system.vals) for system in systems]
            result["epochs_completed"] = len(search.pareto_history)
        else:
            models, result["fit_designs"] = fit_fixed(case, search, spec)
        from epde.structure.weak_rows import active as weak_rows_active
        result["active_evaluation_form"] = "weak" if weak_rows_active() else "strong"
        if result["active_evaluation_form"] != spec["evaluation_form"]:
            raise RuntimeError("requested evaluation form was not active in the EPDE fit")
        for model_id, equations in enumerate(models):
            if not all(getattr(eq, "weights_final_evald", False) and
                       getattr(eq, "fitness_value", None) is not None and
                       np.isfinite(eq.fitness_value) and eq.fitness_value < 1e7
                       for eq in equations):
                continue
            reports = [equation_report(eq, case) for eq in equations]
            result["models"].append({"model_id": model_id, "equations": reports,
                "training_selection_score": float(np.mean([e["fitness"] for e in reports])),
                "heldout_rollout": rollout_report(reports, case)})
        if result["models"]:
            selected = min(result["models"], key=lambda m: m["training_selection_score"])
            result.update(status="ok", selected_model_id=selected["model_id"],
                          selection_rule="minimum mean training fitness; holdout never used for selection")
        else:
            result.update(status="no_candidate", reason="EPDE accepted no fitted candidate for this stage")
    except Exception as exc:
        result.update(status="error", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    result["elapsed_seconds"] = time.perf_counter() - start
    result["finished_at"] = utc_now()
    return result


def build_matrix(args):
    specs = []

    def add(scenario, stage="fixed", derivative="analytic", noise=0.0,
            seeds=(0,), modes=("mean", "joint"), weighting="samples", **extra):
        for seed in seeds:
            for mode in modes:
                spec = dict(scenario=scenario, stage=stage, derivatives=derivative, noise=noise,
                    seed=seed, multisample_mode=mode, weighting=weighting,
                    rps_mode="assignment" if stage == "assignment" else "plain",
                    evaluation_form="strong", population=args.population, epochs=args.epochs)
                spec.update(extra)
                specs.append(spec)

    if args.suite in ("fixed", "all", "smoke"):
        for name in SCENARIOS[:4]:
            add(name, modes=("joint",) if args.suite == "smoke" else ("mean", "joint"))
        if args.suite != "smoke":
            for name in ("logistic", "coupled", "reaction_diffusion"):
                for noise in (0.0, 0.01, 0.05):
                    add(name, derivative="fd", noise=noise)
            add("contradictory")
            for weighting in ("samples", "points"):
                for density in ("coarse", "refined"):
                    add("density", modes=("joint",), weighting=weighting, density=density)
    if args.suite in ("discovery", "all"):
        seeds = tuple(int(x) for x in args.seeds.split(","))
        for name in ("logistic", "coupled", "reaction_diffusion"):
            add(name, stage="discovery", seeds=seeds)
            add(name, stage="discovery", derivative="fd", noise=0.01, seeds=seeds)
    if args.suite in ("compatibility", "all"):
        for name in ("logistic", "reaction_diffusion"):
            add(name, stage="assignment")
            add(name, modes=("joint",), evaluation_form="weak")
    filters = {"scenario": args.scenarios, "derivatives": args.derivatives, "multisample_mode": args.modes}
    for key, values in filters.items():
        if values:
            accepted = set(values.split(","))
            specs = [s for s in specs if s[key] in accepted]
    if args.noise is not None:
        specs = [s for s in specs if s["noise"] == args.noise]
    for index, spec in enumerate(specs):
        fields = [spec[k] for k in ("scenario", "stage", "derivatives", "multisample_mode", "weighting")]
        fields += [spec["evaluation_form"], spec.get("density", "base"), f"noise{spec['noise']:g}", f"seed{spec['seed']}"]
        spec["run_id"] = f"{index:03d}_" + "_".join(map(str, fields))
    return specs


def selected_model(result):
    return next((m for m in result.get("models", []) if m["model_id"] == result.get("selected_model_id")), None)


def summarize(results):
    statuses = {}
    for result in results:
        statuses[result["status"]] = statuses.get(result["status"], 0) + 1
    density, controls, grouped = {}, [], {}
    for result in results:
        spec, selected = result["spec"], selected_model(result)
        group_keys = ("scenario", "stage", "derivatives", "noise", "multisample_mode", "weighting", "evaluation_form", "density")
        group_id = tuple(spec.get(k) for k in group_keys)
        group = grouped.setdefault(group_id, {"settings": dict(zip(group_keys, group_id)), "runs": []})
        group["runs"].append(result)
        if spec["scenario"] == "contradictory" and selected:
            residuals = [s["relative_l2"] for e in selected["equations"] for s in e["clean_training_residuals"]]
            controls.append({"control": "contradictory_laws", "run_id": spec["run_id"],
                "maximum_clean_training_residual": max(residuals),
                "cannot_fit_both_exactly": max(residuals) > 1e-7})
        if spec["scenario"] == "density" and selected:
            coefficients = selected["equations"][0]["canonical_coefficients"]
            if coefficients:
                density[(spec["weighting"], spec.get("density"))] = coefficients.get("u")
    for weighting in ("samples", "points"):
        if (weighting, "coarse") in density and (weighting, "refined") in density:
            first, second = density[(weighting, "coarse")], density[(weighting, "refined")]
            controls.append({"control": "sampling_density", "weighting": weighting,
                "coarse_coefficient": first, "refined_coefficient": second,
                "absolute_change": abs(first - second),
                "expected_coarse": -2.5, "expected_refined": -2.5 if weighting == "samples" else -3.4,
                "matches_analytic_expectation": abs(first + 2.5) < 1e-7 and
                    abs(second - (-2.5 if weighting == "samples" else -3.4)) < 1e-7})
    summaries = []
    for group in grouped.values():
        runs, setting = group["runs"], group["settings"]
        models = [selected_model(r) for r in runs if selected_model(r) is not None]
        coefficients = [max(e["coefficient_error"] for e in m["equations"])
                        for m in models if all(e["coefficient_error"] is not None for e in m["equations"])]
        support_successes = sum(all(e["support_exact"] is True for e in m["equations"]) for m in models)
        holdout = [float(np.mean([s["relative_l2"] for e in m["equations"] for s in e["holdout_residuals"]]))
                   for m in models]
        rollouts = [m["heldout_rollout"]["relative_l2"] for m in models if m["heldout_rollout"]["relative_l2"] is not None]
        summaries.append({"settings": setting, "run_count": len(runs), "returned_model_count": len(models),
            "exact_support_count": support_successes if setting["scenario"] not in ("density", "contradictory") else None,
            "exact_support_rate_all_runs": support_successes / len(runs) if setting["scenario"] not in ("density", "contradictory") else None,
            "median_max_equation_coefficient_error_if_available": float(np.median(coefficients)) if coefficients else None,
            "median_holdout_residual_if_available": float(np.median(holdout)) if holdout else None,
            "median_rollout_error_if_available": float(np.median(rollouts)) if rollouts else None,
            "error_count": sum(r["status"] == "error" for r in runs),
            "timeout_count": sum(r["status"] == "timeout" for r in runs),
            "no_candidate_count": sum(r["status"] == "no_candidate" for r in runs)})
    return {"status_counts": statuses, "groups": summaries, "controls": controls,
            "interpretation": "Fixed-support fits do not establish structure discovery; all seeds and failed runs are retained."}


def write_csv(path, results):
    fields = ["run_id", "scenario", "stage", "derivatives", "noise", "seed", "multisample_mode",
              "weighting", "evaluation_form", "status", "elapsed_seconds", "variable", "equation",
              "coefficient_error", "support_exact", "mean_holdout_relative_l2", "rollout_relative_l2", "error"]
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            model = selected_model(result)
            for equation in model["equations"] if model else [None]:
                row = {key: result["spec"].get(key) for key in fields if key in result["spec"]}
                row.update(status=result["status"], elapsed_seconds=result.get("elapsed_seconds"),
                           error=result.get("error", result.get("reason")))
                if equation:
                    holdout = [v["relative_l2"] for v in equation["holdout_residuals"] if v["relative_l2"] is not None]
                    row.update(variable=equation["variable"], equation=equation["text"],
                        coefficient_error=equation["coefficient_error"], support_exact=equation["support_exact"],
                        mean_holdout_relative_l2=float(np.mean(holdout)) if holdout else None,
                        rollout_relative_l2=model["heldout_rollout"]["relative_l2"])
                writer.writerow(row)


def source_fingerprints():
    """Hash actual source/config contents, including files not tracked by git."""
    files = sorted(set(REPO.glob("epde/**/*.py")) | set(REPO.glob("epde/**/*.json")) | {Path(__file__).resolve()})
    digests = {str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    aggregate = hashlib.sha256()
    for name, digest in sorted(digests.items()):
        aggregate.update(name.encode() + b"\0" + digest.encode() + b"\n")
    return {"captured_at": utc_now(), "scope": "all epde Python/JSON files and benchmark.py, including untracked files",
            "aggregate_sha256": aggregate.hexdigest(), "files_sha256": digests}


def metadata(args):
    def git(*parts):
        return subprocess.check_output(["git", *parts], cwd=REPO, text=True).strip()
    versions = {}
    for package in ("numpy", "scipy", "scikit-learn", "torch", "epde"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"started_at": utc_now(), "git_head": git("rev-parse", "HEAD"),
            "git_status": git("status", "--short"), "python": sys.version,
            "working_tree_diff_sha256": hashlib.sha256(git("diff", "--binary", "HEAD").encode()).hexdigest(),
            "working_tree_diff_scope": "tracked diff only; source_fingerprints also includes untracked implementation files",
            "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_fingerprints": source_fingerprints(),
            "platform": platform.platform(), "packages": versions,
            "threads": {key: os.environ[key] for key in THREAD_KEYS},
            "argv": sys.argv, "arguments": vars(args), "support_tolerance": SUPPORT_TOL,
            "holdout_policy": "whole independent clean trajectories; never used for fit or model selection",
            "noise_policy": "independent Gaussian observation noise; sigma=noise*std(clean field); FD differentiates noisy observations",
            "fixed_fit_path": "EPDE plain_fit._versions, including production rank/stability/acceptance checks; affine retains production intercept selection",
            "assignment_path": "EPDE SoEqRightPartSelector on the supplied candidate term set",
            "discovery_path": "EpdeSearch.fit with a reproducibly seeded random initial population; no truth candidates inserted"}


def launch(spec, directory, args):
    result_path = directory / (spec["run_id"] + ".json")
    log_path = directory / (spec["run_id"] + ".log")
    result_path.unlink(missing_ok=True)
    timeout = args.discovery_timeout if spec["stage"] == "discovery" else args.fixed_timeout
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", json.dumps(spec),
               "--worker-output", str(result_path)]
    start = time.perf_counter()
    worker_environment = dict(os.environ, PYTHONHASHSEED=str(spec["seed"]))
    try:
        with log_path.open("w") as stream:
            completed = subprocess.run(command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT,
                                       timeout=timeout, check=False, env=worker_environment)
        if result_path.exists():
            result = json.loads(result_path.read_text())
        else:
            result = {"spec": spec, "status": "error", "models": [],
                      "error": f"worker exited {completed.returncode} without a result"}
    except subprocess.TimeoutExpired:
        result = {"spec": spec, "status": "timeout", "models": [],
                  "reason": f"hard per-run limit of {timeout:g} seconds"}
    except Exception as exc:
        result = {"spec": spec, "status": "error", "models": [], "error": str(exc)}
    result["wall_seconds_including_imports"] = time.perf_counter() - start
    result.setdefault("elapsed_seconds", result["wall_seconds_including_imports"])
    result["log_file"] = str(log_path)
    result["timeout_seconds"] = timeout
    write_json(result_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("smoke", "fixed", "discovery", "compatibility", "all"), default="smoke")
    parser.add_argument("--output", type=Path, default=Path("projects/multisample/results/benchmark.json"))
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--fixed-timeout", type=float, default=45.0)
    parser.add_argument("--discovery-timeout", type=float, default=90.0)
    parser.add_argument("--scenarios", help="comma-separated scenario filter")
    parser.add_argument("--derivatives", help="analytic,fd")
    parser.add_argument("--modes", help="mean,joint")
    parser.add_argument("--noise", type=float)
    parser.add_argument("--resume", action="store_true", help="retain completed non-error jobs; archive the prior report and retry errors/pending jobs")
    parser.add_argument("--list", action="store_true", help="print the matrix without importing EPDE or fitting")
    parser.add_argument("--check-data", action="store_true", help="check analytic fixture identities only")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        write_json(args.worker_output, run_worker(json.loads(args.worker)))
        return
    if args.population < 2 or args.epochs < 1 or min(args.fixed_timeout, args.discovery_timeout) <= 0:
        parser.error("population>=2, epochs>=1, and positive timeouts are required")
    specs = build_matrix(args)
    if not specs:
        parser.error("the filters selected no runs")
    if args.list:
        print(json.dumps(specs, indent=2))
        return
    if args.check_data:
        checks = []
        for name in SCENARIOS:
            case = make_case(dict(scenario=name, noise=0.0, seed=0))
            evidence = fixture_evidence(case)
            if "maximum_exact_law_residual" in evidence:
                assert evidence["maximum_exact_law_residual"] < 1e-12, (name, evidence)
            if name == "reaction_diffusion":
                assert [x["rank"] for x in evidence["individual_designs"]] == [1, 1]
                assert evidence["pooled_design"]["rank"] == 2
            checks.append({"scenario": name, "evidence": evidence})
        print(json.dumps(finite_json(checks), indent=2, allow_nan=False))
        return
    output = args.output.resolve()
    directory = output.parent / (output.stem + "_runs")
    directory.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": SCHEMA_VERSION, "metadata": metadata(args), "matrix": specs, "results": []}
    pending = specs
    if args.resume:
        if not output.exists():
            parser.error("--resume requires an existing output report")
        previous = json.loads(output.read_text())
        previous_specs = {s["run_id"]: s for s in previous["matrix"]}
        if any(previous_specs.get(s["run_id"]) != s for s in specs):
            parser.error("--resume requires the same matrix and search settings")
        attempt = 1
        while output.with_name(f"{output.stem}_attempt_{attempt}.json").exists():
            attempt += 1
        archive = output.with_name(f"{output.stem}_attempt_{attempt}.json")
        for prior in previous["results"]:
            if prior["status"] == "error" and prior.get("log_file"):
                old_log = Path(prior["log_file"])
                if old_log.exists():
                    archived_log = old_log.with_name(f"{old_log.stem}.attempt_{attempt}.log")
                    old_log.rename(archived_log)
                    prior["log_file"] = str(archived_log)
        write_json(archive, previous)
        retained = [r for r in previous["results"] if r["status"] != "error"]
        retained_ids = {r["spec"]["run_id"] for r in retained}
        report["results"] = retained
        report["previous_attempts"] = previous.get("previous_attempts", []) + [{
            "report_file": str(archive), "retained_run_ids": sorted(retained_ids),
            "source_benchmark_sha256": previous["metadata"].get("benchmark_sha256"),
            "note": "Completed non-error results retained; prior errors remain in the archived report."}]
        pending = [s for s in specs if s["run_id"] not in retained_ids]
    write_json(output, report)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        jobs = {executor.submit(launch, spec, directory, args): spec for spec in pending}
        for future in as_completed(jobs):
            result = future.result()
            report["results"].append(result)
            report["results"].sort(key=lambda r: r["spec"]["run_id"])
            report["summary"] = summarize(report["results"])
            write_json(output, report)
            write_csv(output.with_suffix(".csv"), report["results"])
            print(f"[{len(report['results'])}/{len(specs)}] {result['spec']['run_id']}: "
                  f"{result['status']} ({result['elapsed_seconds']:.2f}s)", flush=True)
    report["metadata"]["finished_at"] = utc_now()
    write_json(output, report)
    print(f"JSON: {output}\nCSV: {output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
