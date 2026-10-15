# Visitor-CPU execution — existing Streamlit demo

The application remains at https://epde-explorer-demo.streamlit.app and continues
using free Streamlit Community Cloud. For native computation, visitors start a
small Python worker on their own computer. The browser talks directly to its
loopback endpoint. There is no paid compute hosting and no second website.

## Simplest Windows x64 setup (one user-approved download)

A native launcher is provided at
[EPDE-Start.cmd](../native_launcher/EPDE-Start.cmd). A GitHub Actions
Windows runner separately tests its bootstrap before it is offered as a
release asset. Use the public
[release page](https://github.com/MakSoS1/EPDE/releases/tag/epde-native-launcher-v1)
**only after that release exists and the verification reports success**.

Download the launcher and explicitly open it. It downloads verified uv
0.13.0, automatically installs Python 3.12, fetches a fixed repository
source revision, then installs the locked EPDE/PySINDy scientific
dependencies into your own user-profile directory. The code used for
equation search is the original EpdeSearch pipeline, not EvoSparse.
Initial setup downloads a large Python/PyTorch/data environment and can
take some time. The launcher opens the existing website and displays
its random local worker access code in the console.

It does **not** bypass a browser's security model or silently execute
a program from the website. It targets Windows x64, not macOS or Linux,
and requires the user's informed approval to run downloaded code.
An unsigned batch file may be restricted by enterprise policies.

## Alternative developer installation (Windows, macOS, Linux)

Install Git and uv (https://docs.astral.sh/uv/getting-started/installation/).
Open Windows PowerShell, macOS Terminal, or a Linux shell, then run:

~~~bash
git clone --branch demo/streamlit-community-cloud https://github.com/MakSoS1/EPDE.git
cd EPDE/projects/pic
uv sync --locked --extra cpu --extra app
uv run --locked --extra cpu --extra app python app/local_worker_service.py
~~~

The worker prints a randomly generated access code. Keep the terminal running.
On the website open Lab, select **My computer**, enter the code, choose EPDE
and/or PySINDy, then **Prepare local runs**. Press **Run on my CPU** inside
the browser component. Paired searches run in sequence and appear in the same
Results/Compare sections when each completes. Ctrl+C stops the worker.

Modern browsers may ask for permission to reach localhost. Allow that
permission for the demo site when prompted. If a corporate browser policy,
privacy extension or mixed-content rule blocks this request, local execution
cannot work without an approved exception. The Demo server option remains
a limited backup, and no browser security policy should be disabled.

## Technical and security boundaries

- The local server binds only to 127.0.0.1:8765, never a LAN interface.
- A unique session code is printed on each restart. Only the demo origin
  and local development origins are allowed through CORS; Private Network
  Access preflight is supported where browsers implement it.
- Only registered scientific methods and data types are accepted. Native
  EPDE/PySINDy execute in a fresh, resource-limited process for up to ten
  minutes. No browser-provided Python scripts, commands or pickle files
  are run. Only one job executes at a time.
- Uploaded files use Streamlit's file uploader before the browser passes a
  numerical NPZ to the local worker. The file therefore ALSO touches the
  free Streamlit server; this build is not a guarantee of private local-only
  data handling.
- Results return as a versioned JSON record to the site's existing
  Results/Compare session storage. This storage is ephemeral: a Streamlit
  Cloud reboot/container change may remove results. Download important JSON.
- The worker uses visitor CPU, NOT CUDA/GPU in this initial version.
- The access code is entered into a Streamlit input, so the site server
  can see the value transiently. It must not be reused as another password.

## Why not completely in browser with no installation?

The real EPDE pipeline uses normal CPython libraries, process isolation
and components not available as equivalent Pyodide/WebAssembly modules.
Pyodide's multiprocessing and subprocess are not functional. A substitute
sparse-regression algorithm written in JavaScript is not the same method.
For that reason native computation requires the visitor to explicitly start
a local worker. Free hosting still serves only the interface and manages
lightweight metadata.

## Testing status

Automated repository CI checks Python/JS syntax, CORS authentication,
scientific configuration safety and a native PySINDy smoke test. Real
browser-to-local-worker communication must additionally be tested with
browser local-network permissions on the target OS and cannot be certified
by server-side unit tests alone.
