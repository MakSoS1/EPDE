# Client-side equation discovery: implementation decision (October 2026)

**Implementation update:** an experimental native loopback worker has now been
added in app/local_worker_service.py, connected via app/support/discovery_lab/
local_browser.py and the Lab's My computer mode. It runs EPDE/PySINDy on the
visitor's CPU. See LOCAL_WORKER_README.md for user setup, trust model, caveats,
and the remaining browser integration tests.


The existing Streamlit deployment executes EPDE and PySINDy on the server.
Ordinary web pages cannot execute CPython processes or access CUDA on the visitor's
operating system without an explicit user-installed helper. Keep this boundary clear.

## Browser-only mode

A WebAssembly/Pyodide implementation could run a light **subset** of numerical
sparse regression in a Web Worker and leave data in the browser. It is **not**
a compatible drop-in replacement for the current EPDE repository. Pyodide
does not provide CPython's ordinary subprocess/multiprocessing, native compiled
PyTorch/CUDA, or the Linux binary wheels used by the server application.
A serious port would need its own package compatibility checks, identical
differentiation algorithms, deterministic seed tests and limits on CPU/RAM.

## Recommended client computation: voluntarily installed worker

Keep the same **Lab / Results / Compare** frontend. Add a deployment-neutral,
versioned job protocol and a user-installed Python worker:

1. The user explicitly installs and starts a signed/reproducible worker on their
   Windows, macOS or Linux PC. The worker reports supported EPDE/PySINDy versions,
   CPU/RAM budget and optional CUDA capability.
2. The user chooses execution **on this computer**, and the UI creates an
   authenticated, single-use job descriptor containing dataset checksum,
   signed validated settings, requested method, resource ceilings, and run ID.
   Never send an arbitrary shell command or unconstrained Python code.
3. The worker **initiates an outbound HTTPS connection** to a small API
   (or long-polls a private job queue), verifies the allowed method/data/settings
   and runs the ordinary benchmark pipeline in a constrained child process.
   This avoids depending on the browser accessing localhost, which can require
   CORS and explicit local-network permissions.
4. The worker sends status, logs and the standardized result record back to
   the API. The result viewer already accepts the same JSON schema.
5. An object store holds immutable input blobs/results, and a database holds job
   ownership, attempts, checksums, worker/software versions and expiry. A single
   Streamlit Community Cloud container's local files are not durable storage.
6. The worker never accepts scripts, unsafe pickle blobs, unknown binaries,
   arbitrary remote URLs, model/LLM API keys or third-party datasets by default.
   Users must opt into downloading public built-in data, and their local private
   input should stay on their machine unless explicitly exported.
7. Quotas, job signing/expiry, server revocation, CPU/RAM/clock limits, authenticated
   result access, worker upgrades and privacy notices are mandatory before
   making the feature public.

Do **not** enable browser-to-localhost requests automatically. On modern
browsers, requests from an HTTPS site into loopback/LAN address space can require
Local Network Access permissions and still require CORS and secure contexts.

## Matching the scientific benchmark

The local and server engines must use the same source revision, data hash,
derivative estimator, token library, random seed and objective definitions.
Record OS, Python/NumPy/BLAS and package versions. Any speed comparison
across machines should report hardware and threading separately.

## Current state

The initial local worker, loopback JSON API and browser-to-worker
Streamlit v2 component **are implemented in the demo branch**. It is a
first experimental integration, not a certified cross-browser release:
there has not yet been a real Chrome/Firefox/Edge + desktop worker end-to-end
test on the deployed Streamlit Cloud website. Server fallback still exists
and executes on the Cloud VM. Local-worker mode executes native searches
on the user's computer only after the user explicitly installs and starts it.
