# Multisample iteration 2: independent audit and before/after notebook

## Requested outcome

Continue the existing multisample work, investigate omissions with another round of fixes and tests, inspect the user's Streamlit-demo fork for useful benchmark coverage, and deliver an executed notebook that makes the effects of the changes directly assessable.

## Baselines and scope

- Upstream before: `8690035e90b234b34ed5a2fb0bb8e4a814dcbdf7`.
- First iteration: `9b9cd7ceafeadd5e9f77e80bf21087fc9089d4ae`.
- Demo source for audit: `demo/streamlit-community-cloud`, `90d99884d32fc0de14ffe42a08cb544bbfa59e13`.
- Continue only `research/multisample-2026-11-13`; preserve upstream and the deployed demo branch. No PR or messages to collaborators.
- Five historical solver-pin failures and three whole-repository collection errors remain known baseline outcomes; do not weaken tests to conceal them.

## Approaches considered

1. Plot only the earlier stored mean/joint results. Cheap, but it cannot demonstrate historical source bugs or verify that the notebook still runs.
2. Run a large independent discovery campaign in every notebook cell. Stronger evidence, but routine notebook execution becomes slow and hard to inspect.
3. Execute a bounded genuine historical/current comparison in separate processes, embed readable results, and expose heavier catalog experiments through an explicit runner. This is the implementation choice: the notebook is practical while the detailed evidence remains reproducible.

## Contracts

- Historical before/after uses actual different source roots. Mean versus joint on one source revision is labeled an ablation, never a historical comparison.
- Every subprocess records source identity/fingerprint, configuration, seed and input case. No accidental import of the active checkout in the baseline worker.
- Input data and coefficients retain physical units; held-out whole trajectories never enter fitting or model selection.
- Observed data should remain coherent across strong and weak paths after ordinary caller-array edits. Snapshot ownership is explicit, including aligned raw and masked static-token values.
- Every accepted domain mask must preserve the declared grid geometry; equal row counts alone do not establish coordinate alignment.
- Numerical robustness fixes must distinguish a stable coefficient helper from an end-to-end search guarantee at extreme floating-point scales.
- Existing demo generators may be reused with pinned provenance, but their historical sparse-regression results are not presented as EPDE results. Historical preprocessing that crosses validation boundaries must not be copied into the new comparison.
- Notebook outputs include failures and limits, confidence is not inferred from a few related synthetic tasks, and positive complementary-excitation examples are accompanied by already-identifiable two-mode controls.

## Deliverables

An executed Russian `.ipynb`, a readable HTML export, a reproducible comparison helper, source changes with red/green regressions, additional actual-EPDE benchmark evidence from demo scenarios, and a report with exact verification outcomes. All deliverables are versioned in the existing research branch.
