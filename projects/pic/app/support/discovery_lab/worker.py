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
sys.path[:0] = [str(APP), str(PIC), str(REPO)]
from support.discovery_lab import contract, queue


def _inside(path, base):
    p, b = Path(path).resolve(), Path(base).resolve()
    if not p.is_relative_to(b):
        raise ValueError("Data/config must belong to this session")
    return p


def _run_record(req):
    from epde_bench.config import variants
    from epde_bench.runner import run_one
    from epde_bench.datasets import REGISTRY
    method = req["method"]
    dataset, variant = req["dataset"], req["variant"]
    if dataset not in REGISTRY or variant not in variants():
        raise ValueError("Unregistered record/variant")
    if variants()[variant].get("method", "epde") != method:
        raise ValueError("Configuration does not belong to selected method")
    seed, noise = int(req.get("seed", 0)), float(req.get("noise", 0))
    if not 0 <= seed <= 999 or not 0 <= noise <= 10:
        raise ValueError("Seed or noise outside demo limits")
    return run_one(dataset, variant, noise, seed)


def _run_upload(req, folder):
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
                          "axis_names": list(problem.axis_names), "truth": None},
              "source": {"kind": "uploaded_npz", "sha256": identity["data_sha256"]}}
    search_cfg = resolve_for_problem({"search": settings["search"]}, problem)
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
        record["metrics"] = metrics.score_run(front, objectives, [])
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
        if method == "llm":
            record = _run_llm(job, folder)
        elif job["source"] == "record":
            record = _run_record(job)
        else:
            record = _run_upload(job, folder)
        record = contract.normalize(record, method)
    except Exception as exc:
        # Never write repr/traceback containing remote request URLs/credentials.
        record = contract.normalize({"status": "error", "error": f"{type(exc).__name__}: method execution failed"}, method)
        print(traceback.format_exc(), file=sys.stderr)
    record["wall_seconds"] = round(time.perf_counter()-start, 4)
    queue._write(folder / "result.json", record)
    return 0 if record["status"] in ("ok", "unsupported") else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
