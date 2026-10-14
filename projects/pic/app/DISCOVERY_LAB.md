# Equation Discovery Lab — development branch

**Branch:** `feature/discovery-lab-demo-2026-10-14` in `MakSoS1/EPDE`  
**Based on:** `research/nir1-mdl-factorial-2026-10-14`, source revision `0127710ed80d41df1aac514ea1619bca6b266b57`.

The development branch adds an opt-in Streamlit research demo for running EPDE and PySINDy on built-in data, uploaded numerical arrays and a restricted scalar ODE generator. It also defines a transport and result contract for Yaroslav's future private LLM agent. It does not replace the scientific EPDE implementation or benchmark protocol and does not include the closed LLM model.

No pull request or deployment is required to work with this branch. Existing research pages remain intact. In demo mode the legacy pages are hidden because the original app has globally visible job records.

## Start locally

From the repository root (Python 3.11–3.13, `uv` installed):

```bash
cd projects/pic
uv sync --locked --extra cpu --extra app
EPDE_PUBLIC_DEMO=1 uv run --locked --extra cpu --extra app streamlit run app/Home.py --server.address 127.0.0.1
```

Windows PowerShell:

```powershell
cd projects/pic
uv sync --locked --extra cpu --extra app
$env:EPDE_PUBLIC_DEMO = "1"
uv run --locked --extra cpu --extra app streamlit run app/Home.py --server.address 127.0.0.1
```

For local research pages without the visitor-only navigation, leave `EPDE_PUBLIC_DEMO` unset.

Docker Compose (requires local Docker resources):

```bash
docker compose -f projects/pic/app/compose.demo.yaml up --build
```

Then visit `http://127.0.0.1:8501`. The Compose port is bound only to localhost.

## Components

| File | Purpose |
| --- | --- |
| `app/Home.py` | Navigation; public mode exposes only Discovery Lab |
| `app/views/10_Discovery_lab.py` | Single method/run interface, plots, history, export |
| `app/support/discovery_lab/contract.py` | Versioned result contract |
| `app/support/discovery_lab/ode.py` | Restricted AST-based ODE builder |
| `app/support/discovery_lab/uploads.py` | Numerical-only, bounded CSV/NPY/NPZ ingestion |
| `app/support/discovery_lab/queue.py` | Capability-scoped jobs, quotas and private paths |
| `app/support/discovery_lab/manager.py` | CPU process supervisor with timeouts |
| `app/support/discovery_lab/worker.py` | EPDE, PySINDy and LLM service adapters |
| `app/Dockerfile.demo` and `app/compose.demo.yaml` | Local container configuration |
| `tests/test_discovery_lab.py` | Isolated contract, equation and queue tests |

## Scientific principles

A benchmark-quality comparison needs the same input arrays, grids, noise realization, boundary cropping, differentiation strategy and declared budgets. EPDE returns multiple candidates and configurable multiobjective scores; the current PySINDy baseline returns a single selected candidate and objectives (held-out normalized MSE and active term count). **Do not combine their scores into one Pareto front or publish a cross-framework ranking based on these unrelated numbers.**

The UI provides a descriptive pair of EPDE/default and PySINDy/pysindy runs on the same dataset/seed/noise or uploaded dataset with shared EPDE differentiation settings. This is an exploratory smoke test, **not** a complete multi-seed benchmarking campaign. The existing `projects/pic/bench.py` campaign engine remains available for Vlad's controlled benchmarks.

The result stores the native `front: list[list[str]]`, `objectives`, `selected`, `metrics`, method, settings, timing and method-specific `criterion_labels`. It displays the selected law and candidate scatter, then attempts to reconstruct from data via `epde_bench.reconstruction`. Reconstruction on a fitted record must not be presented as held-out forecasting.

The generated ODE's RHS is shown only as metadata; until it is translated into a canonical EPDE truth representation, the site must not claim structural recovery accuracy.

## LLM service contract (Yaroslav)

The future model stays in Timur's **private** repository. Do not copy its weights, prompts, API keys, or unpublished source into this public EPDE repository.

Set these variables server-side (never from the browser):

- `EPDE_LLM_SERVICE_URL` — configured HTTPS endpoint or approved local HTTP service;
- `EPDE_LLM_SERVICE_TOKEN` — optional bearer credential between services.

The UI sends a bounded numerical sample via POST JSON:

```json
{
  "schema_version": "discovery-demo/1.0",
  "task": "equation_discovery",
  "source": "record",
  "dataset": "ode",
  "seed": 0,
  "sampled": true,
  "data": {
    "fields": {"u": [1.0, 0.9, 0.8]},
    "coordinates": [[0.0, 0.1, 0.2]],
    "axes": ["t"]
  }
}
```

Reply schema example (illustrative numbers, **not actual measured results**):

```json
{
  "status": "ok",
  "front": [["-0.5 * u{power: 1.0} = du/dx0{power: 1.0}"]],
  "objectives": [[0.012, 0.08]],
  "selected": ["-0.5 * u{power: 1.0} = du/dx0{power: 1.0}"],
  "metrics": {"selected_index": 0},
  "criterion_labels": ["validation normalized MSE", "coefficient instability"]
}
```

The remote service must disclose its objective definitions and perform its own coefficient fitting, verification, request/token limits and API-spend accounting. The browser never controls model URLs or arbitrary prompts. A future iterative agent can keep this output contract while extending metadata. The current sample transport does not include complete grid-shape semantics or physical units; agree on them before claiming generic PDE support.

## Limits and security

- Only lightweight built-in records are offered; no Navier–Stokes demo.
- CSV/NPY/NPZ uploads are bounded to 8 MiB compressed and at most 300,000 numerical elements, with checks on expanded NPZ size.
- Each session uses a randomly generated 192-bit recovery capability and its own result tree.
- A local server manager limits concurrent workers (1–2), tasks (3 pending per session, 30 total queued), execution timeout (30–600 seconds), and local LLM request count (3/hour per session, 50/day instance-wide).
- This is a **single-host research demo**, not a production public multitenant service. There is no identity-based authentication, global distributed rate limiting, IP-based flood protection, completed disk garbage collection, audited resource isolation, or cost-enforced LLM gateway.
- Underlying repository is publicly readable; **a feature branch is not a private security boundary**.
- Prefer localhost during development. Before opening it publicly, add reverse-proxy authentication, per-user quotas, retention, RAM/CPU cgroup isolation and an API budget gateway.

## Verification

The development overlay was checked using:

```bash
python -m unittest discover -s projects/pic/tests -p test_discovery_lab.py -v
```

Ten isolated unit tests passed; Python syntax checks passed; process life-cycle smoke test used a **stubbed** scientific runner. Streamlit browser tests and full live EPDE/PySINDy runs on the actual repository were **not** executed in the artifact preparation environment, and should not be claimed as successful.

Recommended next tasks: full Python dependency install and genuine EPDE/PySINDy smoke runs, cross-device browser checks, file-storage cleanup policy, equality of differentiation settings, Vlad's PDEComp comparison validation, then Yaroslav's private-service integration. No work in `main`, `nir1`, ITMO or another repository is implied by this branch.
