# EPDE Explorer — deployment and release checklist

## Correct source for the current minimal six-page Streamlit website

| Setting | Required value |
| --- | --- |
| GitHub repository | `MakSoS1/EPDE` |
| Git branch | `demo/streamlit-community-cloud` |
| Main app file | `projects/pic/app/Home.py` |
| Python interpreter | `3.12` |
| Dependency manifest | `projects/pic/app/requirements.txt` |

The **same** Streamlit interface provides Home, How it works, Run a search,
Results, Compare and Data catalog. A separate JavaScript-only PySINDy site is
neither deployed nor needed for the scientific workflow.

- Small ODE/PDE: hosted EPDE and PySINDy run on the Streamlit server.
- Large PDEs: the UI never attempts to load, preprocess or run huge datasets,
  even when Streamlit itself is launched on a workstation. It exports
  `experiment.zip` and offers the repository ZIP and operating-system-specific
  launcher instructions.
- The user launches the *native scientific process* using `run-portable.cmd`
  on Windows or `bash run-portable.sh` on macOS/Linux.
- The user uploads `experiment-results.zip` on Results and compares only
  identically fingerprinted numerical inputs. Each method stores its own
  scientific options and the full recovered equations.
- An unsupported PySINDy formulation is never replaced with a different PDE.
  The UI offers an EPDE-only shortcut instead.

## Current public deployment

The former public URL is https://epde-explorer-demo.streamlit.app/.
On **2026-10-16** the independent Chromium check still saw the **previous
navigation and UI**, while the actual six-page app passed local GitHub Actions.
Therefore the former public URL is NOT evidence that this Git branch has
been deployed.

Community Cloud follows its originally deployed GitHub coordinates
(owner/repository/branch/entrypoint), which cannot be changed simply by
pushing commits to another branch. To release without risking the old app:

1. Sign in to https://share.streamlit.io/ as the owner of the deployment.
2. Create a second app from repository `MakSoS1/EPDE`, branch
   `demo/streamlit-community-cloud`, entrypoint
   `projects/pic/app/Home.py`, Python 3.12.
3. Use a **new temporary subdomain** for that app. Wait for dependency
   installation and verify that all six pages appear.
4. Check a short oscillator search, Results/Compare, native-only
   Navier–Stokes export, and actual native-result ZIP import.
5. Only after that succeeds, back up anything still needed from the old app
   and decide whether to delete/redeploy the original app and reassign
   `epde-explorer-demo` to the new deployment.
   Deleting an app can lose temporary results, analytics and share settings.

Changing the branch of an existing Streamlit Community Cloud deployment is
not an ordinary toggle in its app Settings. Official Streamlit guidance says
GitHub coordinate changes generally require delete/redeploy.
Do not delete the existing public app before a replacement has passed checks.

Documentation:
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/rename-your-app
- https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/edit-your-app

## Release tests

The workflow `.github/workflows/demo-streamlit-e2e.yml` boots **real**
Streamlit in GitHub Actions with locked scientific dependencies, runs Chromium
and saves snapshots of the six desktop pages and mobile-width views.

The CI gate checks:
- the six page routes and layout, and width overflow in real Streamlit scroll
  containers at 390 px;
- paired hosted EPDE and PySINDy submission, navigation and saved-session reload;
- a native-only heavy PDE that exports the config and offers platform guidance;
- a genuine portable scientific job using the original EPDE/PySINDy Python
  methods, ZIP import through the Results UI, and matching-input Compare.

`.github/workflows/portable-handoff-qa.yml` adds scientific ZIP, tampering,
platform-launcher and source/data fingerprint tests.
`.github/workflows/demo-public-cloud-smoke.yml` independently audits the
**deployed** public site and must not be confused with a passing local CI build.

Scientific interpretation: inputs are compared by hashes of source/data,
loader settings and the noise realization. In-sample solution R² is not
out-of-sample forecast accuracy; EPDE Pareto objectives and PySINDy STLSQ
internal scores are not the same quantity. Native input manifests use
self-reported hashes and are consistency-checked, not recomputed by Cloud.

## Preservation of previous documentation

For historical context only, see
[ARCHIVE_DEPLOYMENT_2026-10-12.md](ARCHIVE_DEPLOYMENT_2026-10-12.md)
and [ARCHIVE_LOCAL_BROWSER_WORKER.md](ARCHIVE_LOCAL_BROWSER_WORKER.md).
Those old browser-bridge instructions DO NOT describe the current UI.
