# Equation Discovery Browser Lab (zero installation)

A separate **static website** using visitors' own CPU. It is deliberately
placed alongside the existing Streamlit app without replacing it or changing
EPDE core code. This project can be hosted on **GitHub Pages or Cloudflare
Pages for free**. The only downloads are the page assets, Pyodide WebAssembly,
scientific packages and the user's own saved output. No compute requests
are sent to Streamlit, Cloudflare Workers or a paid backend.

## What works

- One page with **Lab / Results / Compare / How it works**.
- Automatic Python/WebAssembly loading in a Web Worker, without requiring the
  visitor to install Python, Git, uv, a local helper or an extension.
- The **actual PySINDy 2.1.0** universal Python wheel installed using Pyodide.
  It performs STLSQ on a polynomial library with the same selected derivative
  estimator across compared runs. Failure to load the official wheel is a
  visible error; no imitation is silently substituted.
- Five generated **ODE** examples: exponential decay, logistic growth,
  damped oscillator, Lotka–Volterra, Van der Pol, with ground-truth laws.
- **CSV** inputs: first column is strictly increasing time, 1–4 numeric
  state variables, minimum 80 and maximum 10,000 rows; up to 1 MiB.
- Numeric differentiation with finite differences or Savitzky–Golay.
- A separately labeled *EvoSparse educational demo*, an original small
  evolutionary subset selector. **This is not the EPDE library.**
- Time-tail (last 20%) derivative holdout scores, rendered equations,
  derivative overlay plots and matching-input comparison.
- IndexedDB result history scoped to the browser/site, export as JSON.
  Data are processed in the browser, not uploaded to a compute server.
- Interactive original EPDE LikeC4 architecture, copied from
  app/static/likec4-views.js during build.

## What is NOT implemented or proven

**The real EPDE library has NOT been ported to WebAssembly.** It depends on
native Python execution patterns and packages including PyTorch. The
educational evolution engine deliberately does not claim EPDE's search
operators, token families, objective functions, MOEA/D configuration or
benchmark equivalence. This site cannot reproduce the 43 original problem
records, multidimensional PDE searches, solver backends, arbitrary search
configuration groups or LLM runs. These stay available through the separate
native-Python Streamlit site, which uses server CPU, or an explicitly installed
local worker.

**The numerical scores are derivative-regression scores, NOT future trajectory
forecasts or independent proof of the governing physical law.** The last 20%
of time levels are held out from coefficient fitting, but the trajectory's
data-generation process is not independently split.

Pyodide downloads code and scientific dependencies from an external CDN and
PyPI on the first visit; the website is installation-free but not offline
on its initial launch. Network extensions and aggressive browser privacy
settings can block these requests. Subsequent runtime loading may be served
from the browser cache. Browsers may evict IndexedDB; export results to JSON.

## Running locally as a developer

~~~bash
python -m http.server 8766 --directory projects/pic/browser_demo
~~~

Open http://127.0.0.1:8766/ in Chrome, Edge or Firefox. **This web server
only serves static files**; it does not compute any equations.

The original compiled LikeC4 script is copied during CI. For local testing
of the educational diagrams:

~~~bash
cp projects/pic/app/static/likec4-views.js projects/pic/browser_demo/
~~~

## Testing (developer only)

~~~bash
python -m pip install numpy scipy scikit-learn pysindy==2.1.0
python -m unittest discover -s projects/pic/browser_demo -p test_science.py -v
node --check projects/pic/browser_demo/app.js
node --check projects/pic/browser_demo/science_worker.js
cd projects/pic/browser_demo
npm install
npx playwright install --with-deps chromium
npm run smoke
~~~

Run the static HTTP server in another terminal before the Playwright test.
The browser test **really loads WebAssembly + PySINDy over the network** and
checks scientific output, page refresh persistence and the Compare display.
It is not a mock scientific calculation. The test must pass before the
GitHub Pages deployment job is allowed to start.

GitHub Actions workflow: .github/workflows/browser-demo-pages.yml.

## Deploying for free

1. Open GitHub repository Settings → Pages. Set Build and deployment
   **Source = GitHub Actions**. This account-level action may be required
   before Pages can publish; a repository commit alone cannot configure it.
2. The workflow runs on the demo branch after browser_demo changes. It builds
   only static assets and deploys them under a GitHub Pages address.
3. Alternatively, connect this repository to Cloudflare Pages with output
   directory projects/pic/browser_demo, no build command **provided you
   copy likec4-views.js into the published static assets**. The GitHub Actions
   workflow handles that copy automatically.

No existing Streamlit app URL is silently replaced. Avoid claiming the
Pages site is live until its deployment is confirmed.

## Security

All user input is handled in a browser-side Web Worker. The numerical Python
runner accepts only whitelisted equations/methods, numeric parameters and
CSV data. It does not eval arbitrary Python scripts. Displayed equations
and CSV headers are inserted into DOM using textContent (not innerHTML).
No sign-in is required; Results are local to the browser origin.

## Reproducibility

Run output records the implementation, numeric settings, random seed,
sampling resolution, derivative estimator and the actual polynomial terms.
Identical CSV bytes, noise, seed, degree and derivative estimator share an
input digest for Compare. EPDE-specific scores are not silently interchanged
with PySINDy objective values.
