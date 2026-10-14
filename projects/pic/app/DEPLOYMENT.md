# Streamlit Community Cloud

Public review demo: https://epde-explorer-demo.streamlit.app/

Deployed on 2026-10-09 from `demo/streamlit-community-cloud` in `MakSoS1/EPDE`.
Cloud verification: the short live evolution completed in 2.632 seconds; a
standard seed-0 oscillator search completed with status `ok`, correct selected
terms and 0.58% coefficient error. Its saved record appeared on Results and was
still available after reloading the application. Full cloud campaigns and
large uploads have not been verified.

Deploy the existing application from the GitHub fork. No separate website or
interface is needed. Community Cloud requires the app and its dependencies to
have been pushed to the selected GitHub branch; it cannot deploy local changes.

Use these settings at https://share.streamlit.io:

| Setting | Value |
| --- | --- |
| Repository | `MakSoS1/EPDE` |
| Branch | `demo/streamlit-community-cloud` |
| Main file path | `projects/pic/app/Home.py` |
| Python version (Advanced settings) | `3.13` |

No application secrets are required. `requirements.txt` beside `Home.py` installs
the app dependencies and Linux CPU-only PyTorch. The app and its subprocesses
import EPDE from this repository, preserving the version under review. Notebook
execution dependencies are not needed for the web application.

After the build, verify the application at its public URL:

1. Open Overview, Data sets and Derivatives; check that plots render.
2. On How EPDE works, run the short evolution and inspect the equation and fronts.
3. Run one default oscillator search with one worker. Check its terminal status
   and open the saved result. This tests background processes as well as the UI.
4. Test a small upload and open its saved result.

Community Cloud has limited CPU and memory. Successful local checks do not prove
that a full benchmark fits those limits; validate small runs on the hosted app
before inviting reviewers. Hosted searches run on Community Cloud, not on the
visitor's computer.

Results, uploads and job controls currently belong to the shared application
instance; they are not isolated per viewer. Use public demonstration data. Treat
cloud-generated files as temporary and download results you need to keep before
restarting or deleting the app.

To end the review demo, delete the app from its Community Cloud management menu.
This does not delete the GitHub repository.

Official documentation:

- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies

## 2026-10-12 supervisor-review update — deploy after CI

The scientific and UI changes developed on **research/nir1-mdl-factorial-2026-10-14**
are separate from the original review deployment. Do not deploy from `main`,
`nir1`, or the older `demo/streamlit-community-cloud` unless you deliberately
merge/cherry-pick the reviewed changes there.

To test the amended version yourself in Streamlit Community Cloud:

1. In your Streamlit deployment settings select repository `MakSoS1/EPDE`,
   branch `research/nir1-mdl-factorial-2026-10-14` and entrypoint
   `projects/pic/app/Home.py`.
2. Confirm the GitHub Actions workflow `nir1-streamlit-postcall-qa.yml` passes
   before switching the production deployment branch. Do not claim that GitHub
   page tests establish actual hosted behaviour.
3. Exercise the five scientific flows: Signal check with intercept OFF and ON;
   Derivatives with polynomial degree >= requested derivative order and valid odd
   window; Run a search with 1/x on a **nonzero** spatial coordinate domain;
   SINDy against EPDE on the same record; and YAML import/export roundtrip.
4. Open Results for both methods and save the exported JSON. Test with public,
   non-sensitive data only and compare hosted run settings with those in GitHub.

### Important scope and security warnings

- The `pysindy` adapter is functional and has its own page; there is no LLM
  provider integration, secret key, cost quota, or arbitrary user-submitted
  Python expression in this release. Those require a joint security/service
  design with the LLM and benchmark owners.
- The additional `1/x` term is explicitly allowlisted and is rejected at
  x=0. Do not attempt to work around a singular grid by dividing by an
  arbitrary epsilon without including that modelling assumption in the law.
- Import accepts declarative, size-bounded YAML and known token-family names;
  it does not run source code. Configuration may still request expensive
  calculations. Do not expose open-ended concurrent heavy searches to anonymous
  users on Community Cloud.
- The existing Streamlit Community Cloud demo **does not isolate per-user uploads,
  process control and result files**. This update does NOT magically provide
  authentication, paid-API metering, user identity, multi-tenant isolation,
  or durable storage. Hosting as a public production service requires those
  architectural changes. For now, use a controlled scientific demo only.
- Spectral and ANN differentiation were explicitly deferred in the 12 Oct
  supervisor meeting; neither is advertised as repaired here.
- The `projects/pic` directory name remains for historical import/API
  compatibility; visible product labels use EPDE Explorer. Renaming it would
  break the reproducible benchmark paths and is deliberately deferred.

Regression source: `projects/pic/tests/test_streamlit_supervisor_20261012.py`.
