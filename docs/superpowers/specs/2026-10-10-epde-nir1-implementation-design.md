# EPDE NIR-1: Research Implementation Design

**Date:** 2026-10-10  
**Status:** Approved engineering design (user confirmation on 2026-10-10); implementation underway.  
**Scientific spec:** `NIR1_EPDE_Research_Protocol_2026-10-10.md` (approved research agenda).  
**Target:** The user's fork [`MakSoS1/EPDE`](https://github.com/MakSoS1/EPDE) only; no upstream ITMO commits, PRs, workflow changes, or executions.

## 1. Goal and success contract

Implement a fully reproducible NIR-1 research workspace for evaluating coefficient-instability objectives, adaptive support regularizers, identifiability safeguards, Pareto-front selectors, and computational cost. The project produces both controlled candidate-level experiments and full EPDE runs, preserving every failure, seed and code revision. It should generate a concise review package in English for the research group, plus a short Russian briefing for the user.

The work is considered **engineering-complete** only when all of the following are true:

1. Research code, test fixtures, evaluation campaigns and report generators exist in a distinct branch of `MakSoS1/EPDE`.
2. Tests demonstrate scaling/column-permutation invariance, collinearity/zero-variance diagnostics, no leakage of truth into selector tuning, deterministic run identity, and bounded failure handling.
3. Full PIC baseline and at least one candidate method can be run by the same campaign runner with recorded resource budgets, status and seeds.
4. GitHub Actions has executed the smoke suite and feasible scheduled batches; every reported run has linked workflow evidence or a clear `not_run` status.
5. Actual output files include paired per-system recovery, front-versus-selected gap, confidence intervals, computational cost and illustrative failures; code-only completion is *not* presented as experiment completion.
6. A machine-readable result manifest plus reviewer-ready `REVIEW.md`, `METHODS.md`, `RESULTS.md` and generated plots are available without requiring access to the researcher's remote Windows workstation.
7. Long-running experiments are resumable and auditable, and the default performance optimizations preserve the mathematical calculation and observed model-selection behavior within declared floating-point tolerances. Any approximation that changes fidelity is opt-in, separately measured, and never described as lossless.

Research hypotheses such as a +10 percentage-point gain are acceptance *targets*, not guarantees: a falsified hypothesis is retained as a documented negative result.

## 2. Discovery and important branch decision

Repository access was verified with the authenticated GitHub account `MakSoS1`; the fork is public and writable by that account. Key branch refs observed on 2026-10-10:

- `pic/review-ready`: `bd6babc8d2cf9b69bcb878ead7048a962ab08cf8`.
- `domain_refactor`: `0f59348a347a2414d709ae5d62050939efc94951`.
- `main`: `7d8b9d31ab50ec7aad00778c8cab1cc0eb01ea31`.

`pic/review-ready` is exactly 12 commits ahead of `domain_refactor`; `main` and `pic/review-ready` have diverged significantly (comparison: review branch 57 ahead, 161 behind). `main` includes extensive `projects/thesis` research, while the review branch has the newer EPDE structure, `projects/pic/epde_bench`, tests and a run/campaign/report pipeline.

Three considered strategies:

| Strategy | Advantage | Cost / risk | Decision |
|---|---|---|---|
| Base on `pic/review-ready`, keep research separate | Latest refactored EPDE + PIC campaign, least impact on existing PR stack | Older `main/projects/thesis` methods require independent adapter/port | **Recommended** |
| Base on `main`, port PIC tooling | Direct access to existing thesis audit scripts | Large rewrite of PIC integration and incompatible API | Reject for NIR-1 |
| Merge `main` and `pic/review-ready` | Single combined history | Large conflict, mixes unrelated research/production changes | Reject |

On approval, create exactly `nir1` from the pinned `pic/review-ready` SHA, without touching other branch refs. If the base branch has moved, compare the new SHA and check for changes in interfaces before creating the branch; do not silently rebase or merge onto a different base.

## 3. Existing interfaces and integration limits

Use existing packages and wrapper conventions; avoid copying all of EPDE:

| Source | Existing behavior | Design implication |
|---|---|---|
| `epde/operators/common/stability.py` | `GramSetup`, `VaryingCoefSetup`, `cv_scores` | Consider it a baseline; isolate any new diagnostics |
| `epde/operators/common/sparsity.py` | `PhysicsInformedLasso`, `VWSRSparsity`, shared keep-rule table | Existing keep-rule already uses weighted Gram + refits: do not re-implement or present it as new |
| `epde/operators/common/objectives.py` | `Instability`, `Discrepancy`, `Complexity` | Objective may not share the *same* channel in all modes: verify dispatch paths explicitly |
| `epde/interface/search_config.py` | default `second_objective=instability`, `instability_metric=chi2`, `sparsity_cls=vwsr` | This is the authoritative production baseline on review branch |
| `projects/pic/epde_bench/runner.py` | one-run record including `front`, `objectives`, `metrics`, `selected` | Reuse to avoid global EPDE setting leakage |
| `projects/pic/epde_bench/campaign.py` | resumable isolated subprocesses, timeout/error record | Extend, not replace, for production-level runs |
| `projects/pic/epde_bench/metrics.py` | canonical structure and `select_compromise` | Freeze equivalence-class scoring and selector contracts in tests |
| `projects/pic/configs/variants.yaml` | existing default, legacy, cv, knee, PySINDy variants | Extend in separate NIR-1 config layer; never silently redefine existing variant names |
| `main/projects/thesis/stability_audit.md` | historical ~126-cell statistic/keep-rule search, `vclog+swap` 58/76 vs production 45/76 on constructed cases | Treat as historical science; replicate select comparisons on current branch and avoid claims of novel re-discovery |

Important scientific observation: the `main` audit attributes `Var` partly to deterministic model misfit and identifies over-pruning plus a collinearity blind spot; it already proposes `NC-only` for the objective and `spectral-corrected Var + median anchor + log + swap` for the regularizer. New work must improve on or validate this, not just relabel it.

## 4. Research software architecture

Create a bounded research package under `projects/nir1_stability/`:

```text
projects/nir1_stability/
  README.md                     # install, commands, caveats (English)
  METHODS.md                    # registered comparison protocol (English)
  REVIEW.md                     # concise reviewer landing page (English)
  RESULTS.md                    # generated factual summary, includes failures
  configs/
    study.yaml                  # pinned strata, environments and budgets
    splits.yaml                 # development/validation/heldout task split
    method_grid.yaml            # exact estimator × penalty × selector grid
  nir1/
    __init__.py
    design.py                   # X/y/weights and groups data contract
    fixtures.py                 # T1–T8 controlled synthetic cases
    gram.py                     # weighted sufficient-statistics engine
    diagnostics.py              # uncertainty, E_j, Q_j, conditioning
    regularizers.py             # candidate weighted EN, group/swap fallback
    model_scores.py             # fixed-candidate whole-law scores
    selectors.py                # fixed-front data-only selectors
    epde_adapter.py             # separate live EPDE objective/regularizer adapters
    datasets.py                 # PIC dataset + oracle support mapping
    runner.py                   # fixed candidate and end-to-end entrypoints
    campaign.py                 # immutable, sharded and resumable manifests
    metrics.py                  # paired success, CI, failure taxonomy, cost
    reporting.py                # report tables and scientific figures
  tests/
    test_fixture_contract.py
    test_invariances.py
    test_gram.py
    test_diagnostics.py
    test_regularizers.py
    test_selectors.py
    test_identity.py
    test_epde_adapter.py
    test_end_to_end_smoke.py
    test_metrics_reporting.py
  reports/                     # small checked-in rendered summaries
  manifests/                   # frozen inputs and GitHub Actions launch records
.github/workflows/nir1-smoke.yml
.github/workflows/nir1-research.yml
```

The package may be condensed into fewer files if the interfaces stay clear. Avoid changes outside the research directory except a narrowly justified plugin integration and the two workflow YAMLs. Never edit upstream, old benchmark results, or old research reports.

### 4.1. Data contract

`ResearchDesign(X, y, sample_weight, environment_id, token_names, metadata)` validates dimensions, finite values, strictly nonnegative weights, at least one effective sample and deterministic token order. The physical coefficient scale is kept separate from standardized coefficients. Entire trajectories/environments, not shuffled space-time points, are grouped for validation.

`GramBlocks` stores `G=X^T W X`, `b=X^T W y`, scalar weighted moments, and per-environment summaries. Matrix accumulation must be chunk-compatible and agree with direct weighted products within finite-precision tolerance. Complexity must be declared as at least `O(n p²)` for dense Gram, `O(p²)` storage plus block count.

An immutable Gram-cache key contains hashes of the **actual** data/derivative representation, ordered token identities and their parameters, sample weights/mask, normalization, environment partition, floating-point dtype, relevant settings and software revision. Reuse Gram by exact submatrix extraction when only a support subset changes. Changed preprocessing, target, weights or token values must miss the cache. Never persist a cache hit merely based on dataset *name* or shape.

### 4.2. Diagnostics and numerical rules

`compute_diagnostics` returns per-term sample/reference coefficient estimate, robust cross-environment excess variation `E`, rank/identifiability `Q`, uncertainty channel `U`, conditioning flags and an `insufficient_evidence` mask. Initial estimator families: current `chi2`, `vcoef`, `cv` (via EPDE baseline), Var/NC as documented in main audit, proposed identifiability-aware scoring, and an independent blocked-OLS check.

Proposed `Q_j` is the relative norm of the component of standardized column `j` orthogonal to other candidate columns (computed via rank-revealing QR/SVD, not unstable normal equations). Exact duplicates must return `Q≈0`. `E_j=max(D_j−U_j,0)/(a_j²+ε)` applies only with a documented covariance model; when the derivative error violates it, do not label `E` unbiased. Coefficients near zero use a robust denominator floor tied to data scale and a flagged confidence interval, not a hidden arbitrary fixed relative threshold.

### 4.3. Regularizers

Independent variants:

- `r0`: OLS/ridge and no support penalty.
- `r1`: existing production VWSR unchanged.
- `r2`: ordinary LASSO/STLSQ and adaptive LASSO (Zou 2006) baselines.
- `r3`: instability-only weighted L1 (isolates the supervisor's central idea).
- `r4`: proposed identifiability-aware weighted elastic net, bounded weights, warm starts, post-selection refit.
- `r5`: correlation-group or swap-based selection (historical swap baseline first, then justified new group variant).
- `r6`: documented historical `vclog+swap` after *verified* current-branch port; failures remain separate from experimental finalists.

Never select `λ`, weights, or final model using ground-truth support on heldout sets. The tuning rule operates on data-only blocked validation or a pre-registered path rule. Preserve both pre-refit and post-refit coefficients.

### 4.4. EPDE integration

Three evaluation levels must have separate commands and output schemas:

1. `audit`: predefined fixed candidate matrices and constructed `truth+decoys`, scores only; no EPDE optimizer.
2. `front`: reuse a frozen Pareto candidate list from an already recorded run; compare selectors and scores without new evolution.
3. `search`: actual EPDE with switched objective and sparsity components, controlled by research-specific configurations; recording `truth_on_front`, `success_selected` and time/resource budget separately.

**Independent intervention requirement:** the current refactor deliberately couples `objectives.instability_metric` to both the Pareto instability score and the VWSR keep rule. A valid criterion-only vs regularizer-only ablation therefore needs two **explicit, research-only overrides** (one for the objective estimator, one for the regularizer's estimator/policy). Both default to `None`, preserving the existing shared-metric production path. Test all four factorial cells `baseline`, `criterion_only`, `regulator_only`, `combined` on the same candidate with separate recorded channels; changing the objective-only override must leave baseline support selection untouched, and changing the regularizer-only override must leave the baseline objective scorer unchanged on a fixed support.

If the active EPDE interface cannot support an injectable scorer/regulator cleanly, first add a narrow, backwards-compatible extension guarded behind an explicit research setting, with regression tests proving default bit-for-bit-equivalent behavior on fixed fixtures. Do not monkeypatch process-global settings across concurrent experiments; use separate processes as the existing PIC campaign does.

### 4.5. Reporting

Output a per-run JSONL/JSON/CSV index with SHA/config/split/data identities, seeds, status (`ok`,`timeout`,`crash`,`unsupported`), support, candidate/selected correctness, coefficient errors, identifiability flags, wall time and peak RAM. Reports separate all planned runs from successful completions and show both counts.

Generated `REVIEW.md` leads with 3–5 verified conclusions and explicit caveats, methods table, paired effect sizes and 95% CIs, failure examples, artifacts and reproducible commands. Figures: correlation vs false deletion, Allen–Cahn candidate ordering, criterion × regulator ablation, front-vs-selected gap, quality–cost frontier. Any missing data produce `not_run` rather than a fabricated number.

## 5. Experiment protocol and staging

Follow the scientific protocol's T1–T8 controls and H1–H10 hypotheses. Register immutable splits before pilot tuning:

1. **S0 candidate diagnostics:** T1–T8 and PIC examples on development split, including high-correlation, rescaling, low-coefficient, constant-vs-variable and noise-in-both-matrices controls; compare several established metrics plus `E/Q` candidate. First verify old `main` audit claims on its own fixtures; explicitly distinguish their `vcoef` from current `chi2`.
2. **S1 search pilot:** six representative supported tasks × five paired seeds × approximately six methods (roughly 180 method-runs). At this gate, analyze runtime, failure modes and three separate effects (criterion, regulator, selector). Finalists freeze after S1.
3. **S2 paired confirmation:** 8–12 tasks × 20 seeds × baseline + two or three finalists (480–960 EPDE-level runs, adjusted **before launch** based on pilot duration and memory). Keep slow KS and NS strata reported even if they require independent smaller grids. Same optimizer budget and token pools where possible; separate native-vs-matched comparisons.
4. **S3 external validation:** new initial conditions and unseen synthetic systems, variable coefficients, wider pool, 2D stress and limited approved 0.5–2% diagnostic noise; abstention, accuracy and calibration checks.

Each stage has stop criteria; no S2/S3 launch before the previous stage's budget/quality gate. A failed hypothesis results in documented negative evidence, not post-hoc replacing the frozen test set or silently tweaking thresholds.

### 5.1. Statistical methods

Primary end-to-end outcome: exact structural recovery of the selected equation in all supported planned runs, with pre-registered equivalent forms. Primary paired contrast: absolute percentage-point difference against `chi2+vwsr` at identical task/data/optimizer seeds. Secondary: truth-on-front, conditional selection failure, model-term false deletion/inclusion, per-class coefficient error, rank of truth-vs-decoys, new-IC residual and rollout, abstention calibration, time and RSS.

Per-system intervals: Wilson; matched binary methods: McNemar with paired risk-difference CI; aggregated task generalization: cluster/stratified bootstrap by **systems** and trajectories rather than space-time points, clear caveat for small system count. Holm correction on a small preregistered family. Report individual systems, not just a global percentage.

### 5.2. Performance policy: exact-first, approximation separately gated

**Tier E0 — baseline profiling:** capture stage-level CPU time, number of Gram builds, dimensions, memory RSS and the actual number of evaluated candidate equations, before changing runtime paths. Reference `main/projects/thesis/new_vs_legacy.md` identifies `EqRightPartSelector` Gram construction as an old hot path, but confirm this on `pic/review-ready` instead of assuming the old percentage persists.

**Tier E1 — mathematically equivalent optimizations enabled by default:** cache a sufficient-statistics matrix once for a fixed candidate pool/weights/target; select support subsets by slicing; reuse factorizations or update/downdate them when numerically safe; vectorize blocked weighted moments; reuse immutable derivative/token computations only when full content identity matches; use warm starts for convex subproblems only when termination certifies the same optimum to the stated tolerance; shard independent runs across Actions VMs without changing seeds or run semantics. All paths use `float64` where the reference requires it and preserve regularizer selection/refit logic.

**Tier E2 — potentially approximate methods disabled by default:** point/row subsampling, reduced derivative resolution, sketching, low-rank truncation, coarse pools, reduced evolutionary evaluations, early stopping, approximate covariance or limited λ paths. These can alter discovery quality and must be separate ablations with explicit approximation labels and a paired accuracy/runtime Pareto; never substitute E2 results for the full-data baseline.

**Losslessness claim policy:** an algebraic speedup is only called `quality-preserving` after paired equality tests on fixed candidate lists, support paths and full search runs using identical random-number streams. Equality means identical structural support and selected equation on the frozen fixtures, and coefficients/objectives equal to an absolute+relative tolerance recorded in the run manifest. The implementation must detect near-threshold ties or numerically ill-conditioned factorizations and fall back to the reference solve. Because floating-point reduction order may differ, **no universal bitwise identity is promised**. Publish every detected discrepancy and count its effect on exact recovery.

**Performance gates:** provide a `performance_regression.csv` showing baseline vs E1 on `n∈{10⁴,10⁵,10⁶}` and `p∈{20,50,100,200,300}` where RAM permits, with correct/incorrect/undecidable result counts, p50/p95 runtime, max RSS, cache hit rate and solver fallback count. A speedup without quality parity is reported as an approximation, not as a successful E1 optimization. Avoid a giant Cartesian full-EPDE grid until the measured pilot budget is available.

## 6. GitHub Actions architecture and execution control

GitHub's published current standard public `ubuntu-latest` runner is 4 CPU / 16 GB RAM / 14 GB SSD; the maximum run time per job is 6 hours. These constraints are inputs to sharding and OOM guards, not grounds for silently discarding KS/NS failures.

The GitHub connector in this session can read workflows/runs and create commits/refs, but there is **no confirmed workflow-dispatch tool** in its exposed action set; the local `gh` command is absent. Furthermore, `workflow_dispatch` only triggers workflows defined on the default branch, and `main` must remain unchanged.

Therefore recommend a scoped **push-triggered experiment workflow** on `nir1`:

- `nir1-smoke.yml`: `push` on research branch scoped to changed research code and workflow; runs small tests, lint/import and a 1-run end-to-end smoke.
- `nir1-research.yml`: `push` on research branch **only** when an immutable launch file in `projects/nir1_stability/manifests/launch/` is added/modified; matrix sharded by dataset group/seed chunk; explicit `concurrency`, job `timeout-minutes` ≤360, fail-fast false.
- The research workflow accepts typed stage inputs from the launch manifest, with allowed workloads `S0`, `S1`, `S2`, `S3`, a fixed run cap and a verified SHA. It must not execute `S2/S3` until a launch manifest has been separately reviewed.
- Each launch produces an immutable ledger of **all** planned job identities and a stable hash. Split work into bounded `(dataset, method, seed-range)` shards whose estimated upper wall-time stays below the 6-hour runner limit; fail clearly if the pilot estimate makes a shard infeasible. A retry never changes its identity or seed.
- Persist an **atomic local result and checkpoint after every completed run** (not only at the end of a 6-hour job), and enforce a soft shard deadline (e.g. 300 minutes) at least 30 minutes before the Action job hard timeout so `if: always()` artifact upload has time to complete. After the soft deadline, mark remaining IDs `incomplete`, exit normally, and launch a resume shard. A sudden VM loss may lose unuploaded local progress; the authoritative remote ledger must then mark those IDs incomplete and rerun them, never pretend that the lost results were saved. The aggregator reads recovered fragments, deduplicates identical identities by content hash, and rejects conflicting rows rather than choosing the favorable one.
- A restart generates a new launch manifest referencing the previous ledger and artifact hashes. It schedules only unfinished or explicitly eligible infrastructure-failed identities; retries are bounded (at most two for runner/transport failures). An algorithmic failure counts as failure and is never silently retried until success.
- Use `permissions: contents: read`, pinned checkout actions, no privileged secrets, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, CPU package installation from `projects/pic/pyproject.toml`, and a limit of one heavy task per runner if memory use is substantial.
- Results leave the VM via `actions/upload-artifact` with immutable run IDs, per-shard CSV/JSON, logs and plots; an aggregator merges only matching manifests and rejects duplicate/stale identities. Document artifact retention and maintain a verified portable copy of final findings.
- Generated tables must include `planned`, `ok`, `timeout`, `crash`, `unsupported`, `incomplete`, retry count, and totals reconciled against the immutable ledger; no success percentage is calculated over a covertly filtered subset. Distinguish provisional partial summaries from frozen final analyses in the title and manifest.

If GitHub rejects workflow creation by the connected app or pushes do not start Actions, stop at that permission boundary and request the user to run the checked-in workflow from GitHub; do **not** repurpose tokens or modify `main` to bypass the blocker.

## 7. Review-first quality and tests

Tests are written before implementations (RED then GREEN). Most are pure NumPy/scikit-learn tests that can run locally or on standard GitHub CPU; the heavy EPDE integration tests are staged. Critical tests:

1. A term and coefficient transformed by inverse column scaling produce the same unscaled equation and support decision.
2. A permuted token order permutes reported diagnostics/selection identically.
3. Two duplicate columns are flagged not identifiable; no false claim that the unique physical term was recovered.
4. A weak real term in Allen–Cahn is not automatically classified as spurious solely due to high relative coefficient uncertainty.
5. Correlated measurements invalidating iid noise assumptions produce honest reliability/abstention flags.
6. Unknown variable coefficients can be selected in the appropriate model class, while the constant-law mode marks the assumption violation.
7. Fixed-front selector depends only on scores/validation data, never `truth` from benchmark metadata.
8. Existing PIC default `chi2+vwsr` outputs, shape and objective ordering are unchanged under research-code presence.
9. Blockwise Gram equals direct weighted Gram within declared error tolerance.
10. Timeouts, crashes, unsupported baselines and empty fronts all appear in summary counts and confidence intervals.
11. Different code/data/split/seed manifests never collide in cached result names.
12. Reporting on empty or partially completed campaigns yields a valid explicit incompleteness report, not made-up results.
13. Exact Gram-cache subset reuse matches a fresh reference solve; changing weights, target, token order, derivative values or dtype produces a cache miss.
14. An intentionally ill-conditioned Gram or a coefficient near the threshold triggers SVD/reference fallback, with the same support choice and a recorded fallback reason.
15. A run stopped after a completed identity resumes without recomputing it, while incomplete/corrupted artifacts are not counted as completed.
16. Replayed runs retain exact seeds, budgets and identity; a duplicate conflicting JSON row makes aggregation fail loudly.
17. Any downsampled or sketched run is visibly tagged `approximate=true`; it cannot enter the primary exact-success comparison without a separately declared analysis stratum.

The end-to-end claim requires GitHub Actions run IDs and downloaded artifacts (not only green unit tests). Any proof of improvement must list all baseline costs and per-system failure cases.

## 8. Deliverable sequence and human review gates

Split the larger scientific protocol into working, independently reviewable engineering slices:

- **Slice A — reproducible audit platform:** controlled cases, Gram/diagnostics, baseline replay, CLI, CSV/JSON and tests. Allows a legitimate initial report even if full EPDE is expensive.
- **Slice B — adaptive support methods:** at least three regularizers, identifiability guard, ablation and current-branch regression invariance.
- **Slice C — Pareto/search integration:** selectors, objective plugins and end-to-end PIC campaign variants, preserving unchanged production defaults.
- **Slice D — GitHub Actions and scale:** smoke, sharded staged campaigns, memory profiling, captured artifacts.
- **Slice E — evidence and reviewer pack:** paired intervals, scientific figures, concise English review memo and Russian handoff.

Each slice ends with a test pass and an independent commit **on the research branch only**. No PR toward the ITMO remote will be created. A PR within the user's own fork can be prepared as a draft **only on separate instruction**, since the user asked for review artifacts, not publication or merge.

After technical specification approval, create a detailed TDD implementation plan specifying every file, interface, test and run command. Do not launch stages before that plan has been reviewed.

## 9. Non-goals and explicit boundaries

- No production ITMO repository modifications, merges, releases, tag creation, or upstream PRs.
- No overwrite of PIC historical results or thesis audit scripts.
- No pretending `58/76` candidate cases prove EPDE's full search improved.
- No arbitrary new weak-form noise campaign, PINN solver work, KAN/LLM token proposal or generalized optimizer rewrite: these are independent NIR directions.
- No launch of a many-hundred-run campaign solely because code compiles. Pilot feasibility, fixed hypotheses and resource boundaries are mandatory.
- No unconditional promise of faster runtime with literally identical floating-point bits or universal recovery: numerical fallbacks, quality-parity evidence, and explicit negative cases are required.
- No claiming a universally calibrated statistical guarantee under errors-in-variables without assumptions and counterexamples.

## 10. Proposed review decision

**Recommend** base `pic/review-ready` at SHA `bd6babc8`, preserve `main/projects/thesis` as a historical evidence and baseline source, implement slices A–E with test-first commits exclusively to `nir1`, and run Github Actions via branch-scoped push triggers. This minimizes conflicts and makes every experimental result traceable to a code state.
