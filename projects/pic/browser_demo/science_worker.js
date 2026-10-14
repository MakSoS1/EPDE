/* Browser-only scientific worker. No user code is evaluated.
 * If official PySINDy does not load, never substitute another algorithm.
 */
"use strict";
const PYODIDE_HOME = "https://cdn.jsdelivr.net/pyodide/v0.28.3/full/";
let instance = null;
let initialized = null;
let running = false;

function status(stage, message) {
  self.postMessage({type: "status", stage: stage, text: message});
}

async function initialize() {
  if (instance) return instance;
  if (initialized) return initialized;
  initialized = (async () => {
    status("runtime", "Loading Python/WebAssembly runtime (first visit can take a while)…");
    importScripts(PYODIDE_HOME + "pyodide.js");
    const runtime = await self.loadPyodide({indexURL: PYODIDE_HOME});
    status("libraries", "Loading NumPy, SciPy, scikit-learn…");
    await runtime.loadPackage(["numpy", "scipy", "scikit-learn", "micropip"]);
    status("sindy", "Installing official PySINDy 2.1.0 from PyPI…");
    try {
      await runtime.runPythonAsync(
        "import micropip\nawait micropip.install('pysindy==2.1.0')\nimport pysindy"
      );
    } catch (err) {
      throw new Error("Official PySINDy failed to load; no mock has been substituted. " +
                      String(err && err.message || err));
    }
    const response = await fetch(new URL("./science.py", self.location.href));
    if (!response.ok) throw new Error("Cannot load science.py");
    const source = await response.text();
    runtime.FS.writeFile("/home/pyodide/science.py", source);
    runtime.runPython("import sys; sys.path.insert(0, '/home/pyodide'); import science");
    instance = runtime;
    status("ready", "Ready. Genuine PySINDy loaded; only your CPU runs numerical searches.");
    return instance;
  })();
  try {
    return await initialized;
  } catch (err) {
    initialized = null;
    throw err;
  }
}

self.onmessage = async ({data}) => {
  if (!data || data.type !== "run" || !data.job) return;
  if (running) {
    self.postMessage({type:"error", id:data.id, message:"A browser job is already running"});
    return;
  }
  running = true;
  try {
    const py = await initialize();
    const job = data.job;
    if (!["pysindy", "evo_demo"].includes(job.method)) {
      throw new Error("Native EPDE is not available in this zero-install browser build");
    }
    status("compute", "Computing on your CPU with " + job.method + "…");
    py.globals.set("browser_request_json", JSON.stringify(job));
    const output = py.runPython("science.run_job(browser_request_json)");
    const result = JSON.parse(output);
    self.postMessage({type:"result", id:data.id, record:result});
  } catch (err) {
    self.postMessage({type:"error", id:data.id,
      message:String(err && err.message || err).slice(0, 1600)});
  } finally {
    running = false;
  }
};
