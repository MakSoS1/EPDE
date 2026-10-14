# EPDE / PySINDy compatibility and client-CPU proof (2026-10-15)

## Distinguish three different questions

1. **Library availability:** EPDE's actual scientific engine imports torch,
   scipy, numpy and its own evolutionary operators; PySINDy 2.1.0 is a
   pure-Python wheel using scipy/sklearn/numpy, and does not require torch.
   Their full **current** dependencies are declared in ONE
   projects/pic/pyproject.toml and locked in projects/pic/uv.lock.
   Older research frameworks, older PySINDy versions or old containers may
   have different version constraints. Do not infer version incompatibility
   for this checkout from a historical remark about other frameworks.
2. **Equation representation:** The shipped epde_bench.baselines.run_pysindy
   adapter is NOT the complete set of features available in PySINDy.
   It supports chosen time-derivative targets and explicit RHS terms.
   It *explicitly refuses* pde_divide (x*u_t and inverse-x term) and ns
   (spatial continuity constraint), using UnsupportedProblem. It does
   support second time derivatives when present in its configured library,
   as tested on 'ode'. PySINDy's wider package has PDELibrary,
   WeakPDELibrary and implicit SINDyPI, but these are NOT wired into this
   fixed-library benchmark. Never describe these two UnsupportedProblem
   cases as fundamental impossibilities for all PySINDy methods.
3. **Browser vs native execution:** Current Pyodide can load the genuine
   PySINDy 2.1.0 wheel. Measured in real Chromium WebAssembly:
   platform='emscripten', torch_importable=false, epde_importable=false,
   native_subprocess=false (OSError). It cannot run our unchanged EPDE
   with the original torch + process-based stack. The browser site has
   a labeled separate *EvoSparse educational* example, not EPDE.

## Tried client-compute approaches

| Approach | Real user CPU | Unmodified EPDE | User install |
|---|---|---|---|
| Web Worker + Pyodide + official PySINDy | yes | no | none |
| Pyodide + unchanged EPDE/PyTorch | yes | currently fails prerequisites | none |
| 32-bit x86 CheerpX/WebVM Linux virtualization | yes | native 64-bit Torch/Python stack unverified / mismatched | none |
| Loopback native worker from a versioned checkout | yes | YES (uses the original EpdeSearch and benchmark) | Python environment |
| Optional Windows one-click bootstrap to native worker | yes | YES after setup | one user-approved launcher |
| Free Streamlit Cloud fallback | no, server CPU | YES | none |

GitHub Pages publishing requires an admin to enable Pages in
repository Settings → Pages → Source: GitHub Actions. The browser smoke
passes independently of deployment. A public GitHub Pages URL must NOT be
claimed live until the deploy action reports success.

## Authentic native same-environment test

Workflow: .github/workflows/native-framework-compat.yml.

A single locked CPU environment installs epde, pysindy and torch and
executes both run_one("ode", "pysindy", ...) and a bounded
run_one("ode", "default", ...) through the original EpdeSearch; it
records package versions and checks unsupported_reason() for
pde_divide and ns. Results are written to a public verification
branch ci/native-epde-pysindy when the workflow completes.

**Important:** The original runner isolates each benchmark job in its
own *process* because the EPDE package maintains global state. This
isolation does not imply separate dependency containers; the
subprocesses can share the same pinned venv. Do not run different
EPDE jobs concurrently in a single interpreter.

Windows native bootstrap workflow:
.github/workflows/native-one-click-windows.yml.
It separately tests an explicitly user-launched command file on
Windows x64. A successful test can publish a GitHub Release asset
for download. No web page can safely launch this command on a user's
host without the user's explicit download/approval.

## Research comparability

For a fair comparison, keep the same input realization, derivative
preprocessor, library budget, search budget and data split; record
when it is not possible. Numerical Pareto objectives are not
interchangeable (PySINDy returns normalized validation error and
active-term count; EPDE uses its configurable criterion pair).
A baseline returning only one equation must not be interpreted as
a true EPDE Pareto front. First verify independent correct behavior,
then consider paired comparisons with explicit common reconstruction
or trajectory-validation metrics.

The Oct 14 meeting transcript calls for independent framework
execution in each framework's present form and postpones direct
method ranking. The exact historical remark about incompatible
containers is not present in the available meeting transcript and
must not be retroactively attributed to a particular package/version.
