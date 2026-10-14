# EPDE Explorer — full scientific interface

This demo branch keeps **the original EPDE scientific implementation** and the
complete dataset registry: **59 ODE/PDE records** (15 core, 33 extended, four
measured, seven supplementary), not a hand-picked set of five browser examples.

### Open without installing anything

**[Open the complete EPDE Explorer in GitHub Codespaces](https://codespaces.new/MakSoS1/EPDE/tree/demo/streamlit-community-cloud)**

1. Follow the link and confirm creation of a Codespace for the demo branch
   (or use GitHub → Code → Codespaces → Create codespace).
2. Wait for the first installation (Python/PyTorch packages are large). The
   **EPDE Explorer** automatically starts on forwarded port **8501**.
3. Open that port in the browser on a laptop, tablet or phone. The forwarded
   port is private to your GitHub account.

Codespaces executes the original Python/EPDE on the **Codespaces VM**, not on
a visitor's device. Usage requires a GitHub account and is subject to the
account's available Codespaces resources, hours, storage and billing terms.
Very large PDEs may exceed the chosen machine's memory. Some original datasets
are external: a registered dataset is not necessarily a downloaded file.

### Run the complete site on your own computer

**No Git installation is required:** open the repository branch, select **Code → Download ZIP**, extract it, and start the script from the extracted root. Alternatively clone the branch using Git.

| System | Command |
| --- | --- |
| macOS / Linux / WSL | `bash run-epde.sh` |
| Windows | Double-click `run-epde.cmd`, or run it in Command Prompt |

Both scripts bootstrap the official `uv` if absent, then run the locked
Python 3.12 + CPU PyTorch + Streamlit project. The first run needs internet,
sufficient disk space and may take some minutes. No administrator rights or
Docker are required for this particular EPDE+PySINDy environment.

Open **http://127.0.0.1:8501**. The local interface sets
`EPDE_LOCAL_FULL_RUN=1`, so it supports the complete registry and real EPDE
without the conservative free-server compute gate. Loading an enormous record
may consume considerable RAM. Large data are loaded only on explicit request.

For the exact manual invocation, with `uv` already installed:

```bash
uv run --project projects/pic --python 3.12 --locked --extra cpu --extra app streamlit run projects/pic/app/Home.py
```

### Existing free Streamlit Community Cloud site

<https://epde-explorer-demo.streamlit.app/>

The repository's **complete scientific catalogue** is visible in **All datasets**.
Hosted Lab allows only bounded, small original EPDE runs, to protect the shared
free VM. Complex 2D/3D equations including Navier–Stokes stay visible with
their metadata and governing-law declarations; execute them locally or in
Codespaces instead of loading the shared server. The full catalogue is distinct
from the browser-only PySINDy tutorial, which intentionally offers a five-ODE
WebAssembly demonstration.

**Deployment status matters:** a CI green build doesn't establish that the
public Streamlit instance is healthy. The public instance has had a
browser-accessibility issue; confirm its real status in
[GitHub Actions](.github/workflows/demo-public-cloud-smoke.yml) before
advertising it. The standalone browser-only PySINDy demo has independently
passed public-browser checks.

### Science and data

- Full registry: `projects/pic/epde_bench/datasets.py`
- Dataset acquisition notes: `projects/pic/DATASETS.md`
- Equation-specific presets and search configurations: `projects/pic/configs/`
- The public Lab and complete catalogue: `projects/pic/app/`
- More about the native worker: `projects/pic/app/LOCAL_WORKER_README.md`
- Historical PDEComp/native integrations are separate: DeepMoD and DISCOVER
  use their own historically incompatible containers; this launcher does not
  silently replace their original environments.

The model has access to whichever numerical data are actually present in this
checkout. Data that require downloading from original scientific sources must
be acquired separately; registration alone does not create a dataset.
