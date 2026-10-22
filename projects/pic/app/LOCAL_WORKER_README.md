# EPDE Explorer — run a heavy scientific experiment from the browser UI

This is the **current** supported workflow. You do **not** run PySINDy
inside your web browser, and you do **not** connect the Streamlit page to an
open localhost worker. A heavy PDE is configured in the browser but computed
by the original, separate native Python launcher on your own computer.

## Browser configuration

Open **Run a search** in the six-page Streamlit app, then:

1. Select the equation in the full Data catalog (for example,
   Navier–Stokes cylinder wake). The scientific arrays are not loaded online.
2. Configure the chosen method and full scientific options. If PySINDy cannot
   represent the equation, choose the offered **Use EPDE only** action.
3. Download **experiment.zip**. It contains method-specific settings,
   original dataset reference, noise, seed, source fingerprint and checksum.
4. Download the repo ZIP using the adjacent button. It points to the exact
   commit running the demo, not the current tip of a branch. The same archive
   URL and full commit SHA are included in `experiment.zip` (README.txt and
   experiment.json). Use that version to preserve the scientific fingerprint.

5. Extract the repository (if downloaded as a ZIP). Put the downloaded
   **experiment.zip** file in its top-level folder, beside run-portable.cmd
   and run-portable.sh.

## Start the local calculation

**Windows:** double-click `run-portable.cmd` from the extracted repository.
The console remains open with success/error details.

**macOS or Linux:** open Terminal in the extracted repository root and run:

```bash
bash run-portable.sh
```

The launcher checks for uv and installs its official distribution when
necessary. It creates the locked Python 3.12 / CPU PyTorch environment on
the computer. The first run therefore needs network access, time and local
disk space. Running or downloading an executable requires the user's consent;
corporate endpoint policies may block unsigned launchers.

The scientific calculations use the EPDE, PySINDy, DeepMoD and DISCOVER Python entrypoints
in separate processes. `EPDE_LOCAL_FULL_RUN` is only for those scientific
processes; it is **never** an override to load heavy PDEs in Streamlit.

Successful native execution creates `experiment-results.zip` in the
repository root. The website never accesses your machine or its local files
without you explicitly uploading that ZIP.

## Review and compare

Open **Results → Import completed experiments**, upload
`experiment-results.zip`, and press **Import these results**. Then open
**Compare**, choose a shared verified input group and inspect Equations,
Metrics and Solution fit.

Matching titles alone never make two experiments equivalent. The import
validates source/data fingerprints, loader settings, noise/seed and scientific
configuration. Uploaded records are self-reported and cannot be independently
attested without recomputing the large original dataset.

If the original large scientific source data are absent in your checkout,
the native launch fails with an explanatory error rather than inventing
equations. Consult `projects/pic/DATASETS.md` and the catalog's
availability metadata. Recreate/export the config after updating the
scientific source if a manifest-source fingerprint mismatch is reported.

PySINDy, DeepMoD and DISCOVER preserve the selected PDEComp native/fixed
protocol and method-specific settings. Unsupported methods are never
silently substituted.

## Historical protocol

The old local bridge/service and its instructions are kept separately in
[ARCHIVE_LOCAL_BROWSER_WORKER.md](ARCHIVE_LOCAL_BROWSER_WORKER.md).
It is not a visible execution mode in Run a search.
