"""One bounded scientific method invocation. Runs in a manager-owned process."""
from __future__ import annotations
import json
import os
import sys
import time
import traceback
from pathlib import Path

APP = Path(__file__).resolve().parents[2]
PIC = APP.parent
REPO = PIC.parent.parent
# Run as a script, this folder would come first on sys.path and its module
# names could shadow the standard library; import it only as a package.
sys.path[:] = [p for p in sys.path if Path(p or '.').resolve() != Path(__file__).resolve().parent]
sys.path[:0] = [str(APP), str(PIC), str(REPO)]
from support.discovery_lab import contract, job_queue as queue


#: Search budgets offered in the Lab: (population, epochs); None keeps the stored setting.
EFFORTS = {"quick": (8, 3), "standard": None, "thorough": (24, 8), "custom": None}


def _inside(path, base):
    p, b = Path(path).resolve(), Path(base).resolve()
    if not p.is_relative_to(b):
        raise ValueError("Data/config must belong to this session")
    return p


def _run_record(req, *, hosted=False):
    from epde_bench.config import variants
    from epde_bench.runner import run_one
    from epde_bench.datasets import REGISTRY
    method = req["method"]
    dataset, variant = req["dataset"], req["variant"]
    if hosted and method in ("epde", "pysindy"):
        from support.discovery_lab.execution_policy import is_local_only
        if is_local_only(dataset):
            raise ValueError("Hosted EPDE refused: heavy datasets must run locally.")
    if dataset not in REGISTRY or variant not in variants():
        raise ValueError("Unregistered record/variant")
    if variants()[variant].get("method", "epde") != method:
        raise ValueError("Configuration does not belong to selected method")
    seed, noise = int(req.get("seed", 0)), float(req.get("noise", 0))
    if not 0 <= seed <= 999 or not 0 <= noise <= 10:
        raise ValueError("Seed or noise outside demo limits")
    effort = req.get("effort", "standard")
    if effort not in EFFORTS:
        raise ValueError("Unknown search effort")
    # New Lab requests carry their *complete effective scientific settings*.
    # Legacy queued jobs without overrides still use the original effort modes.
    from epde_bench.config import load_config, resolve_for_problem
    from epde_bench.datasets import load
    from support.discovery_lab.lab_settings import validate_public_search
    if "overrides" in req:
        overrides = req["overrides"] or {}
        if not isinstance(overrides, dict) or set(overrides) - {"search"}:
            raise ValueError("Invalid public search overrides")
        if overrides and not isinstance(overrides.get("search"), dict):
            raise ValueError("Search overrides must contain a mapping")
        cfg = load_config(dataset, variant, overrides=overrides)
        problem = load(dataset, **cfg.get("loader", {}))
        validate_public_search(resolve_for_problem(cfg, problem), problem.shape, override=not hosted)
    else:
        overrides = None
        if method == "epde" and EFFORTS[effort] is not None:
            population, epochs = EFFORTS[effort]
            overrides = {"search": {"evolution": {"population_size": population,
                                                  "training_epochs": epochs}}}
    if hosted and method in ("epde", "pysindy"):
        from support.discovery_lab.execution_policy import require_hosted_epde
        # The complete effective settings are re-evaluated independently of
        # the frontend. Legacy requests without overrides are gated too.
        cfg = load_config(dataset, variant, overrides=overrides)
        problem = load(dataset, **cfg.get("loader", {}))
        require_hosted_epde(dataset, problem.shape, len(problem.variables),
                            resolve_for_problem(cfg, problem))
    record = run_one(dataset, variant, noise, seed, overrides=overrides)
    record["effort"] = effort
    return record


def _run_upload(req, folder, *, hosted=False):
    import yaml
    import numpy as np
    from epde_bench import env, metrics
    from epde_bench.baselines import run_pysindy, UnsupportedProblem
    from epde_bench.config import resolve_for_problem
    from support.custom_run import problem_from_npz, custom_identity
    from epde_bench.runner import discover
    import datetime
    root = folder.parents[1]  # owners/<token>/jobs/<id> -> owners/<token>
    data_path = _inside(req["data_path"], root)
    config_path = _inside(req["config_path"], root)
    with np.load(data_path, allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    # Re-check archived data after saving (not just when uploaded).
    from support.custom_run import validate_arrays
    validate_arrays(arrays)
    if sum(a.size for a in arrays.values()) > 300_000:
        raise ValueError("Saved data exceed demo cap")
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(settings, dict) or not isinstance(settings.get("search"), dict):
        raise ValueError("Invalid saved search settings")
    method = req["method"]
    seed, noise = int(settings.get("seed", 0)), float(settings.get("noise", 0))
    if not 0 <= noise <= 10 or not 0 <= seed <= 999:
        raise ValueError("Invalid noise or seed")
    env.seed_everything(seed)
    problem = problem_from_npz(data_path)
    # A generated ODE carries its law; uploaded data have none unless the user gave it.
    truth = settings.get("reference_truth")
    if truth is not None:
        if (not isinstance(truth, list) or not 1 <= len(truth) <= 4
                or not all(isinstance(eq, str) and "=" in eq and len(eq) < 2048 for eq in truth)):
            raise ValueError("Invalid reference law")
        problem.truth = list(truth)
    identity = custom_identity(data_path, settings, noise, seed)
    # The original custom identity assumes EPDE; distinguish SINDy and EPDE
    # even when they use identical data and differentiation settings.
    from epde_bench.identity import _digest
    import hashlib
    identity["discovery_worker_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    identity["job"]["variant"] = method
    identity["job"]["method"] = method
    identity["sha256"] = _digest({k: v for k, v in identity.items() if k != "sha256"})
    record = {"dataset": "custom", "variant": method, "method": method,
              "noise": noise, "seed": seed, "started": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "data_file": str(data_path), "config": settings,
              "data_sha256": identity["data_sha256"], "identity": identity,
              "problem": {"title": problem.title, "kind": problem.kind,
                          "shape": list(problem.shape), "variables": problem.variables,
                          "axis_names": list(problem.axis_names), "truth": truth},
              "source": {"kind": settings.get("origin", "uploaded_npz"),
                         "sha256": identity["data_sha256"]},
              "effort": settings.get("effort", "standard")}
    if settings.get("label"):
        record["problem"]["title"] = str(settings["label"])[:160]
    search_cfg = resolve_for_problem({"search": settings["search"]}, problem)
    from support.discovery_lab.lab_settings import validate_public_search
    validate_public_search(search_cfg, problem.shape, override=not hosted)
    if hosted and method in ("epde", "pysindy"):
        from support.discovery_lab.execution_policy import require_hosted_epde
        require_hosted_epde(None, problem.shape, len(problem.variables), search_cfg)
    record["search_config"] = search_cfg
    data = problem.noisy(noise, seed)
    start = time.perf_counter()
    try:
        if method == "epde":
            _, front, objectives, seconds = discover(problem, search_cfg, data)
        else:
            cfg = {"search": settings["search"], "method": "pysindy",
                   "pysindy": {"differentiation": "epde", "threshold": "auto",
                               "max_terms_product": 2}}
            front, objectives, seconds, info = run_pysindy(problem, data, cfg, seed)
            record["baseline"] = info
        record.update(status="ok", fit_seconds=seconds, front=front, objectives=objectives)
        record["metrics"] = metrics.score_run(front, objectives,
                                              [truth] if truth else [])
        selected = record["metrics"].get("selected_index")
        record["selected"] = front[selected] if selected is not None and front else None
    except UnsupportedProblem as exc:
        record.update(status="unsupported", reason=exc.reason)
    record["total_seconds"] = time.perf_counter() - start
    return record


def _run_llm(req, folder):
    import base64
    import urllib.parse
    import urllib.request
    url = os.environ.get("EPDE_LLM_SERVICE_URL", "")
    if not url:
        return {"status": "unsupported", "reason": "LLM adapter awaits Yaroslav's private service"}
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("LLM service URL is not a valid HTTP(S) endpoint")
    if parsed.scheme == "http" and parsed.hostname not in (
            "localhost", "127.0.0.1", "::1", "llm-agent"):
        raise ValueError("Nonlocal LLM service must use HTTPS")
    # An administrator configures the endpoint. Browser input never controls it.
    if req["source"] == "record":
        from epde_bench.datasets import REGISTRY, load
        if req["dataset"] not in REGISTRY:
            raise ValueError("Unregistered dataset")
        problem = load(req["dataset"])
        # Payload is bounded; LLM must know and record that it sees sampled data.
        import numpy as np
        fields = {k: np.asarray(v).ravel()[::max(1, (v.size+1999)//2000)].tolist()
                  for k,v in problem.data.items()}
        coords = [np.asarray(g).ravel()[::max(1, (g.size+1999)//2000)].tolist()
                  for g in problem.grids]
        sample = {"fields": fields, "coordinates": coords, "axes": list(problem.axis_names)}
    else:
        import numpy as np
        root = folder.parents[1]
        data_file = _inside(req["data_path"], root)
        with np.load(data_file, allow_pickle=False) as npz:
            sample = {"fields": {k[4:]: npz[k].ravel()[::max(1,(npz[k].size+1999)//2000)].tolist()
                                 for k in npz.files if k.startswith("var_")},
                      "coordinates": [npz[k].ravel()[::max(1,(npz[k].size+1999)//2000)].tolist()
                                      for k in sorted(npz.files) if k.startswith("axis_") and k != "axis_names"],
                      "axes": list(npz["axis_names"].astype(str)) if "axis_names" in npz.files else []}
    payload = {"schema_version": contract.SCHEMA_VERSION, "task": "equation_discovery",
               "source": req["source"], "dataset": req["dataset"],
               "seed": req.get("seed", 0), "sampled": True, "data": sample}
    key = os.environ.get("EPDE_LLM_SERVICE_TOKEN", "")
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            raise ValueError("LLM redirects are disabled")
    opener = urllib.request.build_opener(NoRedirect)
    request = urllib.request.Request(url, json.dumps(payload, allow_nan=False).encode("utf-8"),
                                     headers=headers, method="POST")
    with opener.open(request, timeout=60) as resp:
        raw = resp.read(512 * 1024 + 1)
    if len(raw) > 512 * 1024:
        raise ValueError("LLM response exceeds 512 KiB")
    reply = contract.llm_payload_valid(json.loads(raw))
    # Remote metadata is not authoritative for local paths or benchmark identity.
    for sensitive in ("data_file", "problem", "config", "search_config", "source", "data_sha256"):
        reply.pop(sensitive, None)
    reply.update(dataset=req["dataset"], variant="llm", seed=int(req.get("seed", 0)),
                 noise=float(req.get("noise", 0)))
    if req["source"] == "upload":
        from support.custom_run import problem_from_npz
        from epde_bench.config import resolve_for_problem
        from epde_bench.identity import _file_hash
        import yaml
        root = folder.parents[1]
        data_file = _inside(req["data_path"], root)
        config_file = _inside(req["config_path"], root)
        settings = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
        problem = problem_from_npz(data_file)
        reply.update(data_file=str(data_file), data_sha256=_file_hash(data_file),
                     config=settings,
                     search_config=resolve_for_problem({"search": settings["search"]}, problem),
                     problem={"title": problem.title, "kind": problem.kind,
                              "shape": list(problem.shape), "axis_names": list(problem.axis_names),
                              "variables": problem.variables, "truth": None})
    else:
        from epde_bench.config import load_config, resolve_for_problem
        cfg = load_config(req["dataset"], "default")
        problem = load(req["dataset"])
        reply.update(config=cfg, search_config=resolve_for_problem(cfg, problem),
                     problem={"title": problem.title, "kind": problem.kind,
                              "shape": list(problem.shape), "axis_names": list(problem.axis_names),
                              "variables": problem.variables, "truth": None})
    return reply


def _sanitize(value):
    """JSON without NaN/Inf: undefined numbers become null."""
    import math
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return _sanitize(value.item())
        except (ValueError, TypeError):
            return str(value)
    return value


def _finite_objectives(record):
    """A candidate whose objectives are not finite keeps its equation, without a point."""
    import math
    if isinstance(record.get("objectives"), list):
        record["objectives"] = [o if isinstance(o, (list, tuple)) and len(o) >= 2 and all(
            isinstance(v, (int, float)) and math.isfinite(v) for v in o[:2]) else None
            for o in record["objectives"]]
    return record


def _thin(values, limit):
    import numpy as np
    values = np.asarray(values, dtype=float)
    step = max(1, int(np.ceil(values.shape[0] / limit)))
    return values[::step]


def _clean(values):
    import numpy as np
    return [None if not np.isfinite(v) else float(v) for v in np.asarray(values, dtype=float).ravel()]


def evaluate(record):
    """How well the selected equation describes the data, method-independently.

    * balance: R^2 between the left-hand side computed from the data and the
      value of the right-hand side, per equation (same derivatives as the search);
    * solution: the equation integrated from the first retained point (ODEs) or
      by the method of lines (explicit first-order-in-time 1-D PDEs), compared
      with the data; thinned series are stored for plots.
    This is an in-sample reconstruction of the record used for the search.
    """
    import numpy as np
    from epde_bench import reconstruction as rc
    from epde_bench.config import load_config, resolve_for_problem
    system = record.get("selected")
    if record.get("status") != "ok" or not system:
        return None
    if record.get("dataset") == "custom":
        from support.custom_run import problem_from_npz
        problem = problem_from_npz(record["data_file"])
    else:
        from epde_bench.datasets import load
        problem = load(record["dataset"], **(record.get("config") or {}).get("loader", {}))
    data = problem.noisy(record.get("noise", 0), record.get("seed", 0))
    search = record.get("search_config")
    if search is None:
        cfg = record.get("config") or load_config(record["dataset"], record.get("variant", "default"))
        search = resolve_for_problem(cfg, problem)
    search = dict(search)
    differentiation = (record.get("baseline") or {}).get("differentiation", "epde")
    if differentiation != "epde":
        search["_reconstruction_differentiation"] = differentiation
    out = {"kind": problem.kind, "variables": list(problem.variables),
           "axis_names": list(problem.axis_names), "balance": {}, "solution": None}
    for eq in system:
        try:
            observed, predicted = rc.equation_fields(problem, eq, search, data)
            out["balance"][eq.split("=", 1)[1].strip()] = rc.r2_score(observed, predicted)
        except Exception as exc:                       # noqa: BLE001
            out["balance"][eq.split("=", 1)[1].strip()] = None
            out.setdefault("notes", []).append(f"balance: {type(exc).__name__}")
    try:
        if problem.dim == 0:
            simulation = rc.simulate_ode(problem, system, search, data)
        elif problem.dim == 1:
            simulation = rc.simulate_pde(problem, system, search, data)
        else:
            simulation = None
    except Exception as exc:                           # noqa: BLE001
        simulation = None
        out.setdefault("notes", []).append(f"solution: {type(exc).__name__}")
    if simulation is None:
        out["solution_note"] = ("This equation cannot be solved here: it is not written as "
                                "'time derivative = terms without it' (EPDE may, for example, keep a "
                                "law multiplied by a common factor such as u_t, or choose a second "
                                "time derivative), or the data have more than one space dimension. "
                                "The equation balance R² above still applies.")
        return out
    solution = {"success": bool(simulation.success), "coverage": float(simulation.coverage),
                "message": simulation.message, "r2": {}, "nrmse": {}}
    if problem.dim == 0:
        t = np.asarray(simulation.t, dtype=float)
        solution["t"] = _clean(_thin(t, 600))
        solution["observed"], solution["simulated"] = {}, {}
        for var in simulation.simulated:
            obs = np.asarray(simulation.observed[var], dtype=float)
            sim = np.asarray(simulation.simulated[var], dtype=float)
            ok = np.isfinite(sim)
            solution["r2"][var] = rc.r2_score(obs[ok], sim[ok]) if ok.all() else None
            spread = float(np.std(obs)) or 1.0
            solution["nrmse"][var] = (float(np.sqrt(np.mean((obs - sim) ** 2))) / spread
                                      if ok.all() else None)
            solution["observed"][var] = _clean(_thin(obs, 600))
            solution["simulated"][var] = _clean(_thin(sim, 600))
    else:
        t = np.asarray(problem.grids[0])[:, 0]
        x = np.asarray(problem.grids[1])[0, :]
        rows = max(1, int(np.ceil(len(t) / 80)))
        cols = max(1, int(np.ceil(len(x) / 80)))
        solution["t"] = _clean(t[::rows])
        solution["x"] = _clean(x[::cols])
        solution["observed"], solution["simulated"] = {}, {}
        for var, sim in simulation.simulated.items():
            obs = np.asarray(data[var], dtype=float)
            sim = np.asarray(sim, dtype=float)
            ok = np.isfinite(sim)
            solution["r2"][var] = rc.r2_score(obs[ok], sim[ok]) if ok.all() else None
            spread = float(np.std(obs)) or 1.0
            solution["nrmse"][var] = (float(np.sqrt(np.mean((obs - sim) ** 2))) / spread
                                      if ok.all() else None)
            solution["observed"][var] = [_clean(r) for r in obs[::rows, ::cols]]
            solution["simulated"][var] = [_clean(r) for r in sim[::rows, ::cols]]
    out["solution"] = solution
    return out


def main(folder):
    folder = Path(folder).resolve()
    job = queue._read(folder / "request.json")
    if not isinstance(job, dict) or folder != queue.job_path(job.get("owner", ""), job.get("job_id", "")).resolve():
        raise ValueError("Invalid request directory")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    method = job["method"]
    start = time.perf_counter()
    try:
        # Opt-in on localhost for full scientific EPDE: hosted mode keeps its
        # strict CPU gate and cannot load expensive 2D/3D PDE input files.
        hosted = os.environ.get("EPDE_LOCAL_FULL_RUN") != "1"
        if method == "llm":
            record = _run_llm(job, folder)
        elif job["source"] == "record":
            record = _run_record(job, hosted=hosted)
        else:
            record = _run_upload(job, folder, hosted=hosted)
        record = contract.normalize(_sanitize(_finite_objectives(record)), method)
    except Exception as exc:
        # Never write repr/traceback containing remote request URLs/credentials.
        record = contract.normalize({"status": "error", "error": f"{type(exc).__name__}: method execution failed"}, method)
        print(traceback.format_exc(), file=sys.stderr)
    record["wall_seconds"] = round(time.perf_counter()-start, 4)
    # Save the search result before the (possibly slow) evaluation: if the time
    # limit stops the evaluation, the discovered equations are not lost.
    queue._write(folder / "result.json", record)
    if record["status"] == "ok":
        try:
            record["evaluation"] = _sanitize(evaluate(record))
        except Exception:                              # noqa: BLE001
            record["evaluation"] = {"error": "evaluation failed"}
            print(traceback.format_exc(), file=sys.stderr)
        record["evaluation_seconds"] = round(time.perf_counter() - start - record["wall_seconds"], 4)
        queue._write(folder / "result.json", record)
    return 0 if record["status"] in ("ok", "unsupported") else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
