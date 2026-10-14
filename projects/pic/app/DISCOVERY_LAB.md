# Equation Discovery Lab — development branch

**Deployed demo source branch:** `demo/streamlit-community-cloud` in `MakSoS1/EPDE`  
**Based on:** `research/nir1-mdl-factorial-2026-10-14`, source revision `0127710ed80d41df1aac514ea1619bca6b266b57`.

The development branch adds an opt-in Streamlit research demo for running EPDE and PySINDy on built-in data, uploaded numerical arrays and a restricted scalar ODE generator. It also defines a transport and result contract for Yaroslav's future private LLM agent. It does not replace the scientific EPDE implementation or benchmark protocol and does not include the closed LLM model.

The existing hosted demo at https://epde-explorer-demo.streamlit.app reads this branch. Its default navigation includes **Run a search**, **Results**, **Compare**, **All datasets** and **How it works**. Legacy research pages remain in the repository but are hidden on the public app because their previous job/result directories were shared between visitors. No pull request to the upstream EPDE repository is required.

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

The public interface combines native EPDE and PySINDy in **Run a search**. There is no execution-location choice or separate PySINDy page in the public navigation. **All datasets** exposes the complete registry. To expose the legacy research navigation **only on a local trusted server**, set `EPDE_SHOW_RESEARCH_PAGES=1` and leave `EPDE_PUBLIC_DEMO` unset. Never expose legacy pages on a public instance.

Docker Compose (requires local Docker resources):

```bash
docker compose -f projects/pic/app/compose.demo.yaml up --build
```

Then visit `http://127.0.0.1:8501`. The Compose port is bound only to localhost.

## Components

| File | Purpose |
| --- | --- |
| `app/Home.py` | Unified Run a search, Results, Compare, complete catalogue and learning navigation |
| `app/views/11_Discovery_results.py` | Current session's history, status and detailed results |
| `app/views/9_How_EPDE_works.py` | Original interactive EPDE stages, LikeC4 diagrams and short optimizer example, plus a PySINDy selector |
| `app/support/sindy_guide.py` | PySINDy architecture using the bundled LikeC4 models, stages, STLSQ and algorithm comparison |
| `app/views/12_Discovery_compare.py` | Matched-input comparison and read-only published archive |
| `app/support/discovery_lab/session.py` | Shared session capability and recovery across all three pages |
| `app/views/10_Discovery_lab.py` | Single run form: EPDE/PySINDy selection, data, presets and paired execution |
| `app/support/discovery_lab/ui.py` | Rendering of one result and of several results side by side |
| `app/support/discovery_lab/contract.py` | Versioned result contract |
| `app/support/discovery_lab/ode.py` | Restricted ODE builder and its law in EPDE form |
| `app/support/discovery_lab/uploads.py` | Numerical-only, bounded CSV/NPY/NPZ ingestion |
| `app/support/discovery_lab/job_queue.py` | Session-scoped jobs, quotas, interrupted-job detection, deletion |
| `app/support/discovery_lab/manager.py` | CPU process supervisor with timeouts |
| `app/support/discovery_lab/worker.py` | EPDE, PySINDy and LLM adapters; evaluation of the selected equation |
| `app/Dockerfile.demo` and `app/compose.demo.yaml` | Local container configuration |
| `app/local_worker_service.py` | Optional native execution on visitor CPU |
| `app/support/discovery_lab/local_browser.py` | Bidirectional browser bridge to loopback |
| `app/support/discovery_lab/local_browser_assets/local_bridge.js` | Local browser fetch + native scientific result transport |
| `tests/test_demo_local_cpu.py` | Loopback authentication and local protocol smoke checks |
| `tests/test_discovery_lab.py` | Contract, generator, queue, real PySINDy runs through the worker, page rendering |

## Using the Lab

1. **Run a search (landing page).** Choose EPDE, PySINDy or both, then choose data: one of 59 registered datasets (ODEs, systems, PDEs,
   the NIR-1 test set) with a known law; a generated ODE `du/dt = f(t, u)` whose law is
   converted to EPDE form so the result can be checked; or an uploaded CSV/NPZ/NPY file
   (examples can be downloaded on the page). Pick one or several methods: they run on
   exactly the same input, noise, seed and derivatives. For generated and uploaded data
   the search space can be set: derivative estimator, highest time/space derivative,
   powers, time as a term, sin/cos of a chosen frequency (allowed as a stand-alone forcing).
2. **Results.** Live status of the searches; for each result the known law next to the
   found equation, a verdict and a term-by-term table, the equation's solution against
   the data, the equation balance, all candidates on the Pareto front, a JSON download.
3. **Compare.** Finished runs are grouped by input; the chosen runs are shown side by side
   (equations, verdicts, a descriptive table, solutions over the data on one figure).

The worker evaluates every selected equation the same way for every method: R² of the
left-hand side computed from the data against the right-hand side (balance), and the
equation integrated from the first retained point (ODEs) or by the method of lines
(explicit first-order-in-time 1-D PDEs, space derivatives up to order four), compared
with the data. Both are in-sample checks on the record the method saw.

The session code is kept in the page address, so a reload or a bookmark reopens the same
experiments; anyone with the address sees them.


## Streamlit Community Cloud update

The hosted entrypoint remains `projects/pic/app/Home.py`; the application URL does not change. The repo `MakSoS1/EPDE`, branch `demo/streamlit-community-cloud`, and deployment settings remain unchanged. The push ordinarily triggers a rebuild. If the old navigation persists, open https://share.streamlit.io, select the existing `epde-explorer-demo` app and use **Reboot**; do not create a second app.

The navigation smoke suite is in `projects/pic/tests/test_demo_workspace_navigation.py`, and `.github/workflows/demo-workspace-qa.yml` is scoped to the demo branch. Historical anonymous run files from the old shared application are deliberately **not** mixed into private session history, because ownership cannot be established; prepublished benchmark campaigns appear separately in Compare.

### Learn how the algorithms work

**How it works** is accessible from the left sidebar. The EPDE tab preserves the original
interactive LikeC4 views, complete stage-selector text, source-code walkthroughs and the
short real evolutionary example. The SINDy tab uses existing LikeC4 views of the shared
benchmark architecture and a SINDy-specific pipeline diagram to explain sparse
regression, time-tail validation and the single-candidate output. Both link back to the
same Lab for running methods; no legacy shared-result pages are reintroduced.

## Scientific configurations in the unified Lab

The **Scientific settings** section is part of Lab, not another launcher. It
uses the old Run a search configuration system, with the 7 EPDE setting groups.
For built-in data it loads:

1. configs/_base.yaml (original shared scientific protocol);
2. configs/<dataset>.yaml (equation-specific requirements, when present);
3. configs/variants.yaml (default, poly, knee, instability CV, legacy, etc.);
4. explicit user edits, recorded as a reproducible search override.

The initial selection is the repository's **record-specific shipped preset**.
It is not claimed to be the empirically *best* algorithm configuration on
every problem or noise level: that claim needs independent matched-budget
multi-seed experiments. A universal profile with conservative 12 × 4 EPDE
budget, bounded derivative orders and adjustable candidate library is used
for uploaded data and generated ODEs.

Lab exposes population, epochs, maximum terms, derivative estimators, polynomial
degree/window, boundary cropping, powers of variables/derivatives, products,
1/x tokens (for safe spatial grids), discrepancy and instability metrics,
second objective, sparsity operator, memory budget, and YAML import/export.
Solver-based fitness and arbitrary operator classes are disabled in the public
Community Cloud demonstration. The full resolved scientific settings travel
with the job manifest to the worker and are checked again there, rather than
being displayed but ignored; the stored run record contains their identity.

**Unified execution (2026-10-15):** In public Streamlit navigation
there is one **Run a search** form and no selector labeled server/local.
Original EPDE and original PySINDy are invoked by the same scientific job queue,
using one input array, noise realization, random seed and explicit search settings.
Their results are stored separately but assigned the same input key for
**Results** and **Compare**. Method objectives are kept distinct and must not
be ranked as directly interchangeable. The short choices (method, group, preset,
derivative type, small result groups) use buttons; long dataset lists remain
searchable.

The free public instance enforces a conservative scientific budget (8000 field
cells, at most two variables, 24 candidates, eight epochs and
cells × candidates × epochs <=250000). Full expensive 2D/3D PDEs such as
Navier–Stokes remain in the 59-record catalogue but are not run on the free
shared VM. The same Streamlit code runs with full local scientific resources
using the root `run-epde.sh` / `run-epde.cmd`, or in private GitHub Codespaces;
these deployment details are in the main README rather than a public mode
switcher.

The standalone five-ODE PySINDy WebAssembly teaching prototype remains
versioned at `projects/pic/browser_demo/`, but is not a separate run page or
a replacement for the original PySINDy adapter in this UI. DeepMoD and
DISCOVER still require their original independent native-container test environments.

History survives navigation and F5 **while the Streamlit instance stays alive**;
its per-session filesystem is not permanent Cloud storage and Community Cloud
restarts may erase it. Download valuable JSON results.

## Scientific principles

A benchmark-quality comparison needs the same input arrays, grids, noise realization, boundary cropping, differentiation strategy and declared budgets. EPDE returns multiple candidates and configurable multiobjective scores; the current PySINDy baseline returns a single selected candidate and objectives (held-out normalized MSE and active term count). **Do not combine their scores into one Pareto front or publish a cross-framework ranking based on these unrelated numbers.**

The UI provides a descriptive pair of EPDE/default and PySINDy/pysindy runs on the same dataset/seed/noise or uploaded dataset with shared EPDE differentiation settings. This is an exploratory smoke test, **not** a complete multi-seed benchmarking campaign. The existing `projects/pic/bench.py` campaign engine remains available for Vlad's controlled benchmarks.

The result stores the native `front: list[list[str]]`, `objectives`, `selected`, `metrics`, method, settings, timing and method-specific `criterion_labels`. It displays the selected law and candidate scatter, then attempts to reconstruct from data via `epde_bench.reconstruction`. Reconstruction on a fitted record must not be presented as held-out forecasting.

A generated ODE receives a structural benchmark truth only when the restricted expression is successfully translated into an EPDE-compatible representation; unsupported expressions must be evaluated against their data without claiming structural recovery.

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

```bash
python -m unittest discover -s projects/pic/tests -p test_discovery_lab.py -v
```

16 tests, including real PySINDy searches through the worker (an ODE, a fourth-order PDE,
a generated ODE scored against its law) and the page rendered on their results. Real EPDE
and PySINDy searches were also run through the queue and the manager on built-in ODEs,
systems and PDEs, generated ODEs and an uploaded CSV.

An earlier version of the module that holds the queue was named `queue.py`: started as a
script, the worker then shadowed the standard library `queue` and every search failed at
import. It is now `job_queue.py`, and scripts drop their own folder from `sys.path`.
