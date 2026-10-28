# Data sets, runs and benchmark for EPDE

This project collects the data sets of the repository in one place and gives one way
to work with all of them: load a record, check that its known law is visible in the
data, run the search, score and inspect the result, compare methods. It works the same
on Windows, Linux and macOS, from the command line, from the notebooks and from a
visual app.

What it consists of:

- **the full registry of 59+ records**: synthetic ordinary and partial differential
  equations with known laws and measured records (pendulums, a robot arm, a ball on a
  beam, sea surface temperature). Each record comes with its grid, its known law and
  the equivalent forms of that law that also count as correct;
- **layered settings**: one shared protocol with a single broad token pool for every
  problem, a short file per record with only what that problem needs, method variants
  for comparisons, and changes given at run time;
- **a command line, short scripts per data folder, notebooks and an app**, all on top
  of the same code;
- **a benchmark**: series of runs in parallel processes that resume after an
  interruption, with reports and confidence intervals.

The library itself and the group's original research scripts are not changed; all of
this lives next to them.

## Streamlit: three-step wizard and appearance (16 October 2026)

The public **Run a search** page is a true three-screen wizard, not a
long simultaneous settings form:

1. **Methods & data.** Choose EPDE, PySINDy, DeepMoD and/or DISCOVER, a built-in equation or an
   uploaded/generated dataset, noise and seed. For known equations the exact
   reference law and a numerical preview are **always visible**.
2. **Settings.** Configure the selected method's actual independent scientific
   controls. The original dataset-specific EPDE YAML stays the default. Expert
   derivative/token/operator settings remain accessible only when needed.
   **Back** preserves prior selections.
3. **Review & run.** Inspect the **full effective scientific YAML** for each
   selected method on separate tabs, confirm the configuration, then start
   hosted search or download the portable experiment for a heavy equation.
   Submitted jobs report running/completed states on the same screen.

The sidebar **Appearance** selector switches between **Light** and **Dark**
without affecting the underlying scientific input. The same theme-aware
native Streamlit components render on desktop and mobile.

There are no hidden governing-law or data-preview expanders in Run a search,
and the chosen Data catalog governing equation is shown directly. Results
shows its import and discovered equations; charts, Pareto candidates and
the actual scientific settings occupy distinct simple tabs.

## Unified browser-to-native scientific workflow (October 2026)

The public interface has **one Run a search page**, with independent EPDE,
PySINDy, DeepMoD and DISCOVER scientific controls.

- Small ODE/PDE: the Streamlit **server** computes the selected supported methods.
  No browser CPU/WebAssembly execution and no desktop installation
  is required for ordinary visitors.
- Heavy PDE (e.g., Navier–Stokes): the shared server never loads or runs it.
  Visitors configure the supported methods and protocols before downloading an
  experiment.zip. Unsupported methods are rejected, not substituted.
- Native execution: download the EPDE repository ZIP pinned by the export, extract
  it, put experiment.zip in its root, then double-click run-portable.cmd on
  Windows or use bash run-portable.sh on macOS/Linux. The launcher installs
  the locked Python/PyTorch CPU environment and starts original scientific
  code with an isolated process per method. Internet is needed on first run.
- Import: upload the resulting experiment-results.zip in Results on the
  website, then compare matching runs in Compare.

Compare only groups runs with an identical fingerprint of the input data,
scientific source, loader parameters, noise, seed and noise protocol. It
does not equate datasets by name alone. Imported results have internally
validated metadata but are NOT independently recomputed on the Streamlit
host. PySINDy, DeepMoD and DISCOVER report no EPDE Pareto scores, so their
results are not placed on EPDE's front. Supported equations are checked against the known law
(Hamming distance and the summed relative coefficient error, as defined by
the PDEComp benchmark) and against the data. In-sample solution R2 is not
held-out forecast accuracy. Claims about framework quality require series
over seeds and noise levels; the app summarises them as success rates with
95 % intervals and a noise limit by PDEComp's rule of 3 successes in 5 runs.

PySINDy on the website is the PDEComp benchmark's own PySINDy, run unchanged
from a pinned commit of PDEComp. On PDEComp's datasets (the same files,
windows and grids) it uses PDEComp's settings for them, under the fixed or
the native protocol; on any other data it uses the native protocol with
PDEComp's default settings and is given only the target time derivative.
It covers ODEs, systems of ODEs and scalar PDEs in one space dimension.

**DeepMoD and DISCOVER are also integrated in the website and portable runner.**
Both use PDEComp revision `4abfe4e9eef5263c35ae5672c6ebfb65aabf8365`; their
framework files and revisions are pinned by SHA-256 in
[pdecomp_sources.json](epde_bench/pdecomp_sources.json). The first use downloads
and verifies those sources. Each method runs in a separate CPU process.

| Method | Native protocol (default) | Fixed protocol |
| --- | --- | --- |
| DeepMoD | DeePyMoD neural network, automatic differentiation, stock `Library1D`, sparsity estimation and least-squares constraint. Requires a scalar 1D spatial PDE with a first time derivative target. | DeePyMoD sparse estimation on PDEComp's shared numerical derivatives and candidate library. Supports ODEs, ODE systems and scalar 1D spatial PDEs; 2D support is limited to the PDEComp Navier–Stokes record. |
| DISCOVER | Recurrent controller and symbolic-operator search on scalar ODEs or 1D spatial PDEs. Computes its own finite-difference derivatives and receives no shared feature matrix. | Searches terms from PDEComp's shared numerical derivative and candidate library, with the same scalar ODE/1D PDE restriction. |

DISCOVER in both protocols and DeepMoD fixed require uniformly spaced time and
spatial coordinates. Their pinned numerical derivative routines use one step per
axis, so irregular grids are rejected before fitting starts.
Native DeepMoD receives the actual coordinates and retains irregular-grid support.

The wizard rejects unsupported method/protocol combinations. It does not
silently replace native training with a fixed-library fit. **Quick demo** starts
DeepMoD native training with up to 250 iterations and 1,000 samples, and DISCOVER with up to
1,000 expression samples with a controller batch size of 100. **PDEComp profile**
restores the pinned dataset settings, which can be larger; the displayed controls
remain editable. Hosted DeepMoD and DISCOVER invocations have a 600-second time limit, while
the portable full-run process has no such training time limit. Short smoke runs
check execution and do not establish equation recovery accuracy.

Native DISCOVER can return symbolic expressions outside EPDE's token grammar.
Those expressions and DISCOVER's reward are retained, with structural scoring
and solution reconstruction marked unavailable. Series containing unscored runs
do not report a complete success rate or noise limit. DeepMoD's native search
records automatic differentiation separately from the app's finite-difference
equation-balance evaluation, which uses the supplied coordinate vectors. The
trained network is not exported by the bridge.
EDL is not integrated.

Repository ZIP: https://github.com/MakSoS1/EPDE/archive/refs/heads/demo/streamlit-community-cloud.zip

The manifest requires matching scientific source/configuration hashes. Use the
immutable repository archive linked by the experiment export to reproduce that
configuration; the branch archive above is for starting a new checkout.
The Streamlit instance has temporary storage; retain imported/exported files.

## Installation

### With uv

Choose one PyTorch build and keep it in every command:

```bash
cd projects/pic
uv sync --locked --extra cpu
uv run --locked --extra cpu python bench.py list
uv run --locked --extra cpu python bench.py run ode --noise 0
uv run --locked --extra cpu python -m ipykernel install --user --name epde-pic --display-name "EPDE PIC"
```

For NVIDIA GPUs with CUDA 12.8 use `cu128` instead of `cpu`, both for the installation
and in every run; the two options exclude each other. A run without the option may
remove PyTorch from the environment, so every example names it. The lock file pins all
versions; EPDE is installed in editable mode from this working copy.

For the Streamlit app and the integrated DeepMoD/DISCOVER methods, include the
`app` extra in both installation and execution:

```bash
uv sync --locked --extra cpu --extra app
uv run --locked --extra cpu --extra app streamlit run app/Home.py
```

This installs the compatible TensorFlow/legacy-Keras stack used by DISCOVER.
The locked CPU environment pins PyTorch to the version used by DeePyMoD.

### With pip

Python 3.11–3.13, from the repository root:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install torch                       # or a CUDA build from pytorch.org
pip install -e .
pip install -r requirements.txt         # required: the package declares no dependencies
pip install -r projects/pic/requirements-bench.txt
```

The commands always use EPDE from the working copy they belong to, even when another
copy is installed in the environment.

## Quick start

```bash
python projects/pic/bench.py list                    # every record
python projects/pic/bench.py info burgers            # data, known law, final settings
python projects/pic/bench.py check ac --noise 0,1,5  # is the law visible in the data
python projects/pic/bench.py run burgers --noise 1 --seed 0
python projects/pic/scripts/kdv.py --plot
```

The short scripts accept the same options as the command line (noise, seed, method
variant, changed settings, signal check, plots, output file) and can be started from
any folder.

## Settings

A run is configured in layers; a later layer wins:

1. the **shared protocol**: coordinates and sine/cosine of frequency 2 in every pool,
   powers of the variable up to 3 and of its derivatives up to 2, up to 10 terms of up
   to 2 factors, population 16, 5 epochs, finite differences — the setup of the group's
   former benchmark;
2. the **file of the record**: a larger population for systems, first derivatives for
   first-order systems, smoothing for measured data, special tokens;
3. a **method variant** compared by the benchmark;
4. **changes given at run time**.

Three values are filled for each record automatically: the boundary excluded from
fitting (10 % of every axis), the highest derivative orders (2 in time, 4 in space)
and the dimensionality of the token families.

## Data sets

The full list with classes, shapes, laws and notes is in [DATASETS.md](DATASETS.md):

- **core** — 15 synthetic problems with a known law, the main benchmark tables;
- **extended** — 3 more synthetic problems;
- **real** — 4 measured records;
- **other** — load, but are not benchmarked (no known law, missing files, or too large
  for a routine run).

A new record needs its files in the data folder, a loader that returns grids, fields
and (if known) the law in EPDE's text form, and a short settings file with what
differs from the shared protocol. The documentation and the script for the record are
then regenerated, and the signal check should confirm on clean data that the law holds
before any search is run.

## Signal check

The check evaluates the terms of the known law on the data, prepared exactly as the
search prepares them, and fits their coefficients by least squares. It reports how
much of the left-hand side the true structure explains and how much each term
contributes. A term that explains almost nothing cannot be identified under this
preprocessing. This is a diagnostic of the data, not a proof that no method can
recover the law.

```bash
python projects/pic/bench.py check pend_single --noise 0
python projects/pic/bench.py check ode --noise 0,1,5 --variant poly
```

Notes on particular records:

- **Turbulence slice:** the derivatives and external fields come from the direct
  numerical simulation. Artificial noise is not allowed: noisy fields would be compared
  with noise-free gradients.
- **Sea surface temperature:** by default the largest rectangular ocean region that is
  finite on every day is used; the original box with land masked can be loaded for
  plotting. The governing law is not known.
- **Darcy flow:** the data files are missing; the loader is kept as an unverified
  prototype.

Shapes, finiteness of the data and a non-empty region after the boundary are checked
before every search.

## Comparing methods

Besides EPDE, PySINDy runs exactly as the PDEComp benchmark runs it (variants
for its fixed and native protocols). The project also keeps its earlier
sparse-regression baseline (PySINDy on a library built from the same fields,
derivatives and extra tokens) for the published campaigns; the website does not
offer it. Problems a method cannot represent are reported as unsupported, not as
failures.

```bash
python projects/pic/bench.py campaign --datasets ode,lorenz \
    --variants default,pysindy_pdecomp_fixed --noise 0,1 --seeds 0-2 --workers 2 --name comparison
python projects/pic/bench.py report projects/pic/results/comparison
```

Every run of a campaign is a separate process. Repeating the command continues the same
campaign; failed runs can be retried. A run is identified by its settings, data, code
and dependency versions: changing any of them requires a new campaign, so old and new
results are never mixed.

The report shows the success rates with 95 % Wilson intervals, a ranking by problem
class on the problems every variant supports, and run times. Planned and unsupported
runs are listed separately; errors and timeouts count as failures.

## Notebooks

| notebook | content |
|---|---|
| 00 · Quick start | one complete run, step by step, and how to read the result |
| 01 · Search settings | the search object, its settings and defaults, settings stored in files |
| 02 · Preprocessing | domain, data, derivatives, supplied derivatives, signal check |
| 03 · Search space and fit | token families, the search, the Pareto front and its scoring |
| 04 · Evolutionary optimizer | the evolutionary loop, objectives, sparsity, budget |
| 05 · ODEs | forced, Van der Pol and Duffing oscillators |
| 06 · ODE systems | Lotka–Volterra and Lorenz systems |
| 07 · PDEs in 1-D | equations in one space dimension |
| 08 · PDEs in 2-D and 3-D | equations in two and three space dimensions |
| 09 · Measured data | pendulums, robot arm, ball on a beam, sea surface temperature |
| 10 · Benchmark | comparison of methods |

The tutorials separate data inspection, checks of the known law, discovery and
reconstruction. Where a search is performed, its selected equation is compared with
the data: the left-hand side computed from the data against the value predicted by
the equation and, for ordinary differential equations, the integrated solution
against the record. Reconstruction starts at the first retained interior point.
The oscillator lesson also demonstrates discovery on the first 70% of raw samples
and forecasting the remaining 30%, with candidate selection using training data only.

The notebooks run missing searches themselves and reuse a saved result only when its
settings, data, code and dependencies match exactly. Long searches are not run inside
the notebooks.

## Visual app

A front end to the same code, grouped by step. Every page starts with a minimal set of
controls and its defaults are the stored settings of the record; further choices are under
*More options*, and every page explains itself under *About this page*.

| group | page | what it does |
|---|---|---|
| Start | How EPDE works | the stages of the algorithm and a short live evolution |
| Explore data | Data sets | browse the records, plot them, read the known laws |
| | Signal check | is the known law visible at a given noise level; contribution of every term |
| | Derivatives | derivative methods compared on noisy data, with their error |
| Find equations | Run a search | choose a record and press Start; the result shows the found equation, whether it is the known law, and the equation against the data |
| | Your data | upload a table, a matrix or an archive and search for its equation |
| Review | Results | every saved search: from the app, the notebooks and the campaigns |
| | Benchmark | start a campaign, follow it, read the comparison |

```bash
uv sync --locked --extra cpu --extra app
uv run --locked --extra cpu --extra app streamlit run app/Home.py
```

Ordinary dataset and uploaded-data searches run in separate processes and save the
common benchmark record, so their results can be inspected in a notebook. The short
optimizer and chronological forecasting lessons have their own evidence records;
their computations are isolated from Streamlit sessions as well.

## Checks

```bash
python -m unittest discover -s projects/pic/tests -v
```

Tests cover data, term parsing, metrics, run identity, process lifecycle and small
executable optimizer/API lessons. Full dataset campaigns run separately. Checks that
need files missing from the repository are skipped and reported as such.

### Reading and running in the visual app

The existing Streamlit pages also serve as interactive documentation. Each page includes expandable reading notes explaining the question it answers, how to interpret its output and a small exercise using its existing controls. The algorithm page runs a short real evolutionary search and displays observed epoch fronts. No separate website is required.

From `projects/pic`, start the local application with:

```bash
uv run --locked --extra cpu --extra app streamlit run app/Home.py --server.address 127.0.0.1
```

GitHub Pages cannot serve the Python Streamlit application. Local execution uses the
application above; Community Cloud runs Python on its server. Published experiment
folders retain raw run records, settings, source identities and validation evidence.
Notebook outputs distinguish discovery, reconstruction and unseen-data forecasting;
optional extended comparisons must be enabled explicitly before running them.

The architecture sources are included in `architecture/`; they describe framework packages and the benchmark workflow.
