# Native-method demo review — 2026-10-20

Checked locally on Windows, Python 3.12.7, Chrome/Playwright, with the actual
Streamlit application and its process queue. No deployed Community Cloud
instance was changed or verified in this review.

## Observed execution

Four real searches received the same `nir1_heat` numerical input, noise 0
and seed 0, and completed with `status: ok`:

| Method | Tested budget | Reported fit time |
| --- | --- | --- |
| EPDE | Population 4, one evolution epoch | 3.45 s |
| PySINDy | PDEComp native defaults | 0.19 s |
| DeepMoD | Native Quick demo: 250 iterations, 1000 samples | 93.95 s |
| DISCOVER | Native Quick demo: 1000 expressions, batch 100 | 29.28 s |

Fit times exclude queueing, interpreter startup and result evaluation. The
test confirmed actual native NN/RNN execution and retained equations; these
small-budget runs did **not** recover the exact reference law. Scientific
accuracy is reported separately from successful method execution.

The full native DISCOVER profile also completed (5000 expressions,
47.46 s fit, 82.41 s including startup). Full native DeepMoD on the same
81×96 grid exceeded the hosted 600-second limit. Therefore the hosted form
now offers an explicit Quick demo default and the complete PDEComp profile
as a separate choice. Exported portable runs keep the full profile and do
not have the hosted training deadline.

Local scientific dependencies included NumPy 2.3.4, PyTorch 2.11.0+cu126
(CPU execution), PySINDy 2.1.0, TensorFlow 2.20.0 and tf-keras 2.20.1.
The existing Community Cloud manifest pins PyTorch 2.9; its exact Linux
environment still needs the added CI/deployment checks.

## Browser checks

- Selecting all four methods and the native protocols, editing controls,
  switching full/quick budgets and reviewing effective settings.
- Actual Start → queue → saved result files → Results → Compare.
- Selecting all four results, checking equations, metrics, solution-fit and
  equation-balance tabs, and retaining the session after a reload.
- Desktop 1440×1000 and mobile 390×844, with light and dark appearances.
- No Streamlit exceptions, browser JavaScript errors or horizontal page
  overflow in the completed end-to-end run.

Screenshots and actual scientific result metadata are under
`projects/pic/browser_demo/.cache/native-ui-final/` locally. The CI workflow
runs the same browser script and uploads `native-method-browser-qa` artifacts.
That Linux workflow was added, not executed by this local review.

## Problems fixed during review

- Learning rate 0.002 was rendered as 0.00; floating inputs now preserve it.
- Switching budgets retained old browser widget values; controls now reset
  explicitly, and both browser/AppTest regressions cover the transition.
- Compare accepted only three results; it now accepts four in a two-column
  card grid, with correct per-method budget labels.
- Long equations were clipped. Polynomial equations now wrap; native
  expressions have a safe display parser and a wrapping text fallback.
- The ODE solution plot used the first result's kind even when that result
  had no solution. It now uses the first available solution.
- Review settings now retain the chosen seed and expand native settings in
  portable reviews; a stray question mark in result captions was removed.
- Training timeouts now terminate the child-process tree and produce a
  timeout result with a readable message. A real subprocess regression
  checks that descendant processes stop.

## Automated checks

The full selected suite passed **64 tests**, including actual DeepMoD and
DISCOVER training. After the final display regression was added, the
targeted suite passed **20 tests**. These counts overlap.
Python compilation, JavaScript syntax, `git diff --check` and offline
`uv lock --check` passed. Earlier PySINDy parity checks also passed; the
four-method browser run confirmed its queue and result integration again.

Reproduce the browser check from `projects/pic/browser_demo` with the app
running on localhost:8501:

```powershell
npm ci
$env:PLAYWRIGHT_CHANNEL = 'chrome'
$env:NATIVE_UI_ALL_METHODS = '1'
node smoke_native_methods.mjs
```

Alternatively install Playwright's Chromium and omit `PLAYWRIGHT_CHANNEL`.
