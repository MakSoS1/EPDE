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

**Data catalog** lists all **64 registered ODE/PDE records**. Choosing an
equation starts a fresh experiment in **Run a search**, while retaining saved
results. Registration does not mean every external data file is already
available or every method supports that equation.

**Run a search** has three steps:

1. **Methods & data:** choose **EPDE**, **PySINDy**, **DeepMoD** and/or
   **DISCOVER**, then a built-in equation, uploaded data or a generated ODE,
   with shared noise and seed settings.
2. **Settings:** adjust each selected method's scientific controls and
   protocol. **Suggested** starts from its profile and adapts resource settings
   to the input. **Default** restores EPDE's saved budget; the sparkle
   **Suggest settings** button reapplies editable starting suggestions.
3. **Review & run:** inspect the effective scientific YAML for each method,
   confirm the configuration, then start a supported hosted search or prepare
   a portable experiment for a heavy equation.

Compatibility depends on the data and protocol:

| Method | Supported workflow |
| --- | --- |
| EPDE | Original EPDE search with the record's scientific settings. Hosted ODEs and small 1D PDEs are subject to the shared server's compute limits. |
| PySINDy | PDEComp's pinned implementation for ODEs, ODE systems and scalar 1D PDEs, plus its own Navier–Stokes record. Fixed and native protocols are available on PDEComp records; other supported inputs use native. |
| DeepMoD | Native neural-network/autograd training requires a scalar 1D PDE with a first time derivative target. Fixed-library sparse estimation supports ODEs, ODE systems and scalar 1D PDEs; 2D support is limited to the PDEComp Navier–Stokes record. |
| DISCOVER | Native recurrent-controller symbolic search or fixed-library search for scalar ODEs and scalar 1D PDEs. |

DISCOVER in both protocols and DeepMoD fixed require uniformly spaced time
and spatial axes; native DeepMoD accepts actual irregular coordinates.
Unsupported combinations are rejected. See the
[method protocols and budgets](projects/pic/README.md#unified-browser-to-native-scientific-workflow-october-2026)
for detailed constraints.

Small supported searches execute on the Streamlit server. Heavy records,
including large 2D/3D problems, stay visible in the catalog but are not loaded
or searched by the website. Their **Review & run** screen provides the
portable workflow when a matching repository revision is available:

1. Download **experiment.zip** and **Download EPDE repository ZIP**. The source
   archive is pinned to the exact commit recorded by the export.
2. Extract that source archive and place `experiment.zip` in its root.
   Double-click `run-portable.cmd` on Windows, or run `bash run-portable.sh`
   on macOS/Linux. The first run installs pinned dependencies and needs
   internet access; required external datasets must also be available.
3. Upload the resulting **experiment-results.zip** on **Results**. **Compare**
   groups runs only when their input data, scientific source, loader settings,
   noise and seed identities match.

**Compare → Insights** explains metric-specific leaders and useful next steps
for all four methods. It separates selected-equation recovery, reconstruction,
held-out evaluation and runtime, and states when a comparison is unavailable.
Internal objectives and unavailable scores do not produce an overall ranking.

**Export availability must be checked on the deployed site.** A passing CI
build does not establish that the public download path is working. If the
running checkout cannot be matched to a clean committed revision, export is
withheld and **Export availability details** explains the reason.
The [public Cloud browser check](.github/workflows/demo-public-cloud-smoke.yml)
tests the catalog reset and downloads a Navier–Stokes configuration, recording
the deployed source revision separately from the CI revision.

Results are stored on the running instance under a private session code,
without a durable external database. Navigation and reloads can recover the
same results through the session URL or saved code while those files survive.
Anyone with that link or code can access the session; Community Cloud restarts
may clear its files. Download important JSON results from **Results** and
keep locally generated result archives.

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
