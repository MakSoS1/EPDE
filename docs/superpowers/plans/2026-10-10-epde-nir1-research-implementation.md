# EPDE NIR-1 Research Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a reproducible coefficient-instability and identifiability-aware adaptive-regularization study in `MakSoS1/EPDE`, using GitHub Actions to produce a documented, auditable reviewer package.

**Architecture:** Start a dedicated research branch from `pic/review-ready` at the pinned reference `bd6babc8d2cf9b69bcb878ead7048a962ab08cf8`. Keep NumPy/SciPy candidate-level analysis independent of EPDE internals; add narrowly scoped compatibility-preserving hooks for full search and reuse the existing PIC benchmark runner. Shard slow campaigns into immutable, resumable, check-pointed GitHub Actions jobs; only analytically equivalent speedups are enabled by default.

**Tech Stack:** Python 3.11–3.13, NumPy ≥2.0, SciPy ≥1.12, scikit-learn ≥1.4, PyYAML, pytest, EPDE in this checkout, existing `projects/pic/pyproject.toml` and `uv.lock`, GitHub Actions (`ubuntu-latest` CPU), Markdown/CSV/JSON/PNG artifacts.

**Spec:** `docs/superpowers/specs/2026-10-10-epde-nir1-implementation-design.md`.

**Research protocol:** `docs/research/NIR1_EPDE_Research_Protocol_2026-10-10.md`, copied unchanged from the approved scientific study before committing implementation.

## Global Constraints

- GitHub repository mutations exclusively in `MakSoS1/EPDE:nir1`, never ITMO, never `main`, `domain_refactor`, or `pic/review-ready`.
- Before creating branch, check SHA of `pic/review-ready`; expected `bd6babc8d2cf9b69bcb878ead7048a962ab08cf8`. If different, inspect modifications affecting this plan; do not silently reset or rebase.
- Keep `projects/pic/epde_bench` data-loading, config, canonical truth, subprocess isolation, logging and report behavior; no renaming production variants.
- The baseline is the review-branch default: `wape + chi2 + vwsr`; `main/projects/thesis` is archival external evidence, not the default codebase.
- All parameter tuning and model selection is data-only on development/validation sets, not test labels.
- Run one campaign unit per `(dataset, variant, noise, data_seed, optimizer_seed, config_sha, code_sha, split_sha)` and preserve every failure/timeout.
- Use full-precision baseline objective, fit/selection and support; no default downsampling, smaller search budget, sketching or silent approximation.
- For exact-first performance, preserve support/selected law on frozen fixtures and coefficients/objectives within stored absolute/relative float tolerances; ill-conditioned or near-threshold cases fall back to reference arithmetic. Do not promise universal bitwise identity.
- Each GitHub-hosted job ≤ 360 minutes, CPU runner memory budget 16 GiB; segment jobs, store per-run artifacts, resume on explicit identities, bounded infrastructure retries.
- All run results and scientific reviewer docs in English; short Russian handoff is an additional generated artifact. Never fill absent results with invented values.
- Stage gates S0 → S1 → S2 → S3; do not launch S2/S3 merely because S1 code passed tests. User's full-research authorization still requires an explicit fixed manifest and a measured feasibility budget.

## Review Focus

These failure classes require deliberately written tests in the owning tasks, even when the main scientific example does not exercise them:

1. **Data leak:** a selector accidentally reads `Problem.truth` or benchmark `truth_alternatives` during inference → Task 8 asserts identical selection after changing only truth labels.
2. **Cache collision:** same-shaped arrays with changed derivatives/weights or changed token order reuse Gram → Tasks 2 and 10 assert misses, value parity and logged fallback.
3. **Correlated/degenerate design:** nearly duplicate columns give a meaningless large instability penalty or unstable numerical support → Tasks 3, 4 and 10 require explicit identifiability flag and reference fallback.
4. **Interrupted Actions job:** 6-hour timeout leaves partial results that are silently omitted or counted as complete → Tasks 9 and 11 assert resume with exact planned IDs and accurate completion fractions.
5. **Metric laundering:** report excludes algorithm errors/unsupported forms or treats truth-on-front as selected success → Task 12 asserts separate denominators, invalid partial claims, and full status reconciliation.

---

## Slice A — Core audit and reproducibility

### Task 1: Branch isolation, source-of-truth metadata and research package contract

**Files:**
- Create: `docs/superpowers/specs/2026-10-10-epde-nir1-implementation-design.md` (bring in approved spec verbatim)
- Create: `docs/research/NIR1_EPDE_Research_Protocol_2026-10-10.md` (bring in approved research protocol verbatim)
- Create: `projects/nir1_stability/README.md`, `projects/nir1_stability/nir1/__init__.py`
- Create: `projects/nir1_stability/nir1/identity.py`, `projects/nir1_stability/tests/test_identity.py`
- Modify: `projects/pic/pyproject.toml` only if necessary to declare pytest dev dependencies; do not change `epde` dependency semantics

**Interfaces:**
- Consumes: actual branch/commit refs and existing PIC `epde_bench.identity` canonicalization conventions.
- Produces: `canonical_run_id(spec: Mapping[str, object]) -> str`, `hash_file(path: Path) -> str`, `verify_git_revision(expected_sha: str, repo_root: Path) -> None`.

- [x] **Step 1: Verify base and create isolated Git branch/worktree.** Confirmed pinned SHA; created local and remote branch `nir1` without touching the base branch. Expected: `git rev-parse HEAD` matches pinned base and branch name matches requested research branch; no upstream mutation.
- [ ] **Step 2: Write `test_identity.py` first.** Assert two field-order permutations hash identically, a changed dataset hash/seed/config changes run ID, and missing essential keys raises `ValueError`. Include test verifying the stored protocol/spec are readable.
- [ ] **Step 3: Run tests RED.** Run `python -m pytest projects/nir1_stability/tests/test_identity.py -q`. Expected: import error or defined assertion failure, not environment/package installation failure.
- [ ] **Step 4: Implement pure identity module and install/test harness.** Use canonical sorted JSON + SHA256; run IDs include code, data and config fingerprints. Keep package imports cheap and free from `epde.globals`.
- [ ] **Step 5: Run GREEN and existing PIC regression smoke.** Run research test plus `python -m pytest projects/pic/tests/test_runner_identity.py -q`. Expected: all pass; capture output.
- [ ] **Step 6: Commit `chore(nir1): isolate research scope and reproducible run identities`** in research branch only.

### Task 2: Weighted sample contract and exact sufficient-statistics engine

**Files:**
- Create: `projects/nir1_stability/nir1/design.py`, `projects/nir1_stability/nir1/gram.py`
- Create: `projects/nir1_stability/tests/test_design_gram.py`

**Interfaces:**
- Consumes: `canonical_run_id` from Task 1.
- Produces: `@dataclass(frozen=True) ResearchDesign(X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray, environment_id: np.ndarray, token_names: tuple[str, ...], metadata: dict[str, object])`; `@dataclass(frozen=True) GramBlocks(G: np.ndarray, b: np.ndarray, yy: float, weight_sum: float, by_environment: dict[str, tuple[np.ndarray, np.ndarray, float]])`; `accumulate_gram(design: ResearchDesign, chunk_rows: int = 8192) -> GramBlocks`; `subset_gram(gram: GramBlocks, indices: Sequence[int]) -> GramBlocks`.

- [ ] **Step 1: Write tests for weighted moments.** For `X=[[1,0],[1,1],[0,1]]`, `y=[1,2,3]`, `w=[1,2,3]`, assert `G == X.T @ diag(w) @ X`, `b == X.T @ (w*y)` and `yy == sum(w*y*y)`; assert chunk sizes 1 and 8192 agree.
- [ ] **Step 2: Write validation tests.** Reject negative/NaN weights, mismatched shapes, duplicate token names, zero effective samples; assert deterministic environment blocks and exact support-subset Gram slices.
- [ ] **Step 3: Run RED.** `python -m pytest projects/nir1_stability/tests/test_design_gram.py -q`. Expected: implementation imports not found.
- [ ] **Step 4: Implement design validation and chunked block accumulation.** Use `float64`, fixed stable token order, `X.T @(w*X)` with no hidden intercept; caller must explicitly supply any constant column, named `__intercept__`.
- [ ] **Step 5: Run GREEN, direct parity and memory sample.** Tests pass; independent `numpy.linalg` calculation agrees within `rtol=1e-11, atol=1e-11` on normal conditioning, record numerical error.
- [ ] **Step 6: Commit `feat(nir1): exact weighted Gram and environment statistics`.**

### Task 3: T1–T8 deterministic scientific fixtures and representability checks

**Files:**
- Create: `projects/nir1_stability/nir1/fixtures.py`, `projects/nir1_stability/nir1/datasets.py`
- Create: `projects/nir1_stability/tests/test_fixtures.py`
- Create: `projects/nir1_stability/configs/study.yaml`, `projects/nir1_stability/configs/splits.yaml`

**Interfaces:**
- Consumes: `ResearchDesign`, PIC dataset loader, PIC truth equivalence metrics.
- Produces: `make_fixture(case_id: str, seed: int = 0) -> tuple[ResearchDesign, dict[str, object]]`; `check_representability(dataset: str, tokens: Sequence[str]) -> dict[str, object]`; `load_registered_split(path: Path, stage: str) -> tuple[str, ...]`.

- [ ] **Step 1: Write fixtures tests.** Assert controlled truth coefficients and support for T1 orthogonal, T2 correlations 0/0.5/0.9/0.99/0.999, T3 scale `1e-4`, T4 competing model, T5 independent initial conditions, T6 varying coefficient, T7 `y`-noise plus `X`-noise, T8 exact duplicate/rank deficient. For deterministic seeds, compare SHA over arrays.
- [ ] **Step 2: Add data-registry tests.** Confirm available PIC IDs and supported truth alternatives from existing registry; split files must be disjoint by *system*, no same-system trajectory in heldout selection tuning.
- [ ] **Step 3: Run RED.** `python -m pytest projects/nir1_stability/tests/test_fixtures.py -q`. Expected: missing fixture API / failing assertions.
- [ ] **Step 4: Implement cases and frozen split definitions.** Use separate independent random generators for signal, derivative error and optimizer seed; do not load old Windows paths. Missing real datasets are marked unavailable, not synthesized and mislabelled as real.
- [ ] **Step 5: Run GREEN and PIC compatibility truthcheck.** Tests pass, and `python projects/pic/bench.py info ac` resolves under the PIC environment.
- [ ] **Step 6: Commit `test(nir1): preregister deterministic identification controls and splits`.**

### Task 4: Identifiability, uncertainty decomposition and equation score

**Files:**
- Create: `projects/nir1_stability/nir1/diagnostics.py`, `projects/nir1_stability/nir1/model_scores.py`
- Create: `projects/nir1_stability/tests/test_diagnostics.py`, `projects/nir1_stability/configs/method_grid.yaml`

**Interfaces:**
- Consumes: `ResearchDesign`, `GramBlocks`, T1–T8 fixtures.
- Produces: `@dataclass(frozen=True) Diagnostics(beta: np.ndarray, excess_variation: np.ndarray, expected_variance: np.ndarray, identifiability: np.ndarray, reliable: np.ndarray, rank: int, condition: float, warnings: tuple[str, ...])`; `estimate_diagnostics(design: ResearchDesign, *, ridge: float = 1e-8, rtol: float = 1e-10) -> Diagnostics`; `score_equation(diagnostics: Diagnostics, *, aggregator: str = 'upper_quantile') -> float`.

- [ ] **Step 1: Write RED tests.** Scale a single column and transform physical coefficient inversely: normalized diagnostics preserve score/support; permute columns and compare reordered result; duplicate exact column returns `identifiability≈0` and `reliable=False`, not false certainty; noise in design triggers reliability warning.
- [ ] **Step 2: Write controlled variance tests.** Under iid analytic noise, calibrate `expected_variance`; under correlated block errors, label assumption violated, never return an unjustified 95% CI; T6 varying coefficient gets explicit `model_class='variable'` flag.
- [ ] **Step 3: Run RED.** `python -m pytest projects/nir1_stability/tests/test_diagnostics.py -q`. Expected: missing APIs.
- [ ] **Step 4: Implement QR/SVD residualization, per-environment coefficient estimates and `E=max(D−U,0)/(a²+eps)` with scale-documented denominator; use data-only model score aggregation and reasoned warnings.** Avoid a universal statistical guarantee when both `X` and `y` are estimated derivatives.
- [ ] **Step 5: Run GREEN plus T1–T8 audit CLI smoke.** `python -m pytest .../test_diagnostics.py .../test_fixtures.py -q` passes; stage S0 CLI includes provenance and negative controls.
- [ ] **Step 6: Commit `feat(nir1): identifiability-aware stability diagnostics`.**

## Slice B — Adaptive selection and controlled alternatives

### Task 5: Reference sparse-regression baselines and adaptive weighted elastic net

**Files:**
- Create: `projects/nir1_stability/nir1/regularizers.py`, `projects/nir1_stability/tests/test_regularizers.py`

**Interfaces:**
- Consumes: `ResearchDesign`, `Diagnostics`, Gram from Tasks 2–4.
- Produces: `@dataclass(frozen=True) SparseFit(beta: np.ndarray, support: np.ndarray, objective: float, converged: bool, kkt_residual: float, fallback_reason: str | None, iterations: int)`; `fit_weighted_en(design: ResearchDesign, *, l1: float, l2: float, weights: np.ndarray, tol: float = 1e-8, max_iter: int = 2000, warm_start: np.ndarray | None = None) -> SparseFit`; `build_penalty_weights(diagnostics: Diagnostics, strategy: str, gamma: float, bounds: tuple[float,float]) -> np.ndarray`; `post_refit(design: ResearchDesign, support: np.ndarray, ridge: float = 0.0) -> np.ndarray`.

- [ ] **Step 1: Write RED tests for L1/L2 optimization.** On diagonal Gram compare the closed-form soft threshold, on correlated small design compare SciPy bounded reference objective and KKT, on duplicated design require bounded output and explicit fallback. Compare `weights=ones` with unweighted LASSO baseline.
- [ ] **Step 2: Write truth-protection tests.** T3 weak true term and T2 correlation: no implicit hard-delete by `Q≈0`; `w_min≤w_j≤w_max`; validate no truth labels or oracle coefficients enter `build_penalty_weights`; refit leaves nonselected coefficients zero.
- [ ] **Step 3: Run RED.** `python -m pytest projects/nir1_stability/tests/test_regularizers.py -q`. Expected: missing implementation.
- [ ] **Step 4: Implement coordinate descent on weighted Gram, correct `1/(2n)` normalization, bounded weights, KKT stopping and SVD/ridge fallback; add classic adaptive LASSO, instability-only and `E/Q`-aware strategies.** Ensure `λ` chosen by data-only blocked validation, not exact support.
- [ ] **Step 5: Run GREEN and SciPy cross-check.** Tests pass, KKT finite; record `converged=False` as a failure, never return silent best-effort success.
- [ ] **Step 6: Commit `feat(nir1): adaptive penalty path and reference regression baselines`.**

### Task 6: Correlation-aware swap/group remedies and historical `vclog+swap` baseline

**Files:**
- Create: `projects/nir1_stability/nir1/correlation.py`, `projects/nir1_stability/nir1/historical.py`, `projects/nir1_stability/tests/test_correlation.py`

**Interfaces:**
- Consumes: `SparseFit`, `ResearchDesign`, existing main-branch documented numerical recipes as **reference only**.
- Produces: `correlation_groups(design: ResearchDesign, threshold: float = 0.9) -> list[tuple[int, ...]]`; `swap_refine(design: ResearchDesign, fit: SparseFit, *, tie_rel: float = 0.01) -> SparseFit`; `historical_vclog_weights(design: ResearchDesign, diagnostics: Diagnostics) -> np.ndarray`.

- [ ] **Step 1: Write RED tests.** Reproduce T2, T5 and `burgers_inviscid` collinear twin controls; swap corrects a survivor/pruned candidate only if normalized RSS improves or within 1% with lower predeclared complexity, never by seeing truth; no changing a unique obvious token under orthogonal design.
- [ ] **Step 2: Write historical parity fixture.** Implement identical, separately labelled historical Var-only spectral factor/median/log/swap equations from `main/projects/thesis/stability_audit.md`; parity test requires original controlled support decisions where source fixtures are available, otherwise marks `archival_unreproduced`.
- [ ] **Step 3: Run RED.** `python -m pytest projects/nir1_stability/tests/test_correlation.py -q`. Expected: missing APIs.
- [ ] **Step 4: Implement grouped support and swap with conditioning safeguards; implement and label historical adapter.** The old `58/76` count stays a historical reference until rechecked, not a regenerated result.
- [ ] **Step 5: Run GREEN.** Tests pass and difference report includes both original reported outcomes and current-branch replay statuses.
- [ ] **Step 6: Commit `feat(nir1): correlation-group recovery and historical comparison`.**

### Task 7: Frozen-candidate evaluation and independent score audit

**Files:**
- Create: `projects/nir1_stability/nir1/audit.py`, `projects/nir1_stability/nir1/cli.py`, `projects/nir1_stability/tests/test_audit.py`

**Interfaces:**
- Consumes: fixtures, model scores, regularizers and historical adaptation.
- Produces: `evaluate_candidate_set(case_id: str, methods: Sequence[str], seed: int) -> list[dict[str, object]]`; CLI `python -m nir1.cli audit --cases T1,T2,T3,T4,T5,T6,T7,T8 --seeds 0-4 --output projects/nir1_stability/reports/s0/fixtures.jsonl`.

- [ ] **Step 1: Write RED tests.** For a saved candidate pair, score without optimizer; rank truth/decoys separately from support recovery; identical seed and config yield identical row identity; changing only truth label never changes algorithm score.
- [ ] **Step 2: Run RED.** `python -m pytest projects/nir1_stability/tests/test_audit.py -q`. Expected: missing CLI/evaluator.
- [ ] **Step 3: Implement row schema with `dataset_sha`, `method`, `assumptions`, `truth_status`, `rank`, `warnings`, `elapsed`, `run_id`; reject NaN score unless explicitly marked invalid.** Use identity from Task 1, no user workstation paths.
- [ ] **Step 4: Run GREEN and S0 sample.** `python -m pytest .../test_audit.py -q` passes and `audit --cases T1,T2 --seeds 0-1` produces 4×method rows with no fabricated full-search metrics.
- [ ] **Step 5: Commit `feat(nir1): auditable fixed-candidate experiment runner`.**

## Slice C — Pareto and actual EPDE discovery

### Task 8: Fixed-front data-only selectors and full-search integration

**Files:**
- Create: `projects/nir1_stability/nir1/selectors.py`, `projects/nir1_stability/nir1/epde_adapter.py`, `projects/nir1_stability/tests/test_selectors.py`, `projects/nir1_stability/tests/test_epde_adapter.py`
- Modify only if necessary: `epde/interface/search_config.py`, `epde/operators/common/objectives.py`, `epde/operators/common/sparsity.py`, `epde/operators/common/survival.py`
- Create: `projects/nir1_stability/configs/epde_variants.yaml`

**Interfaces:**
- Consumes: `score_equation`, `fit_weighted_en`, PIC `run_one`, PIC `metrics.select_compromise`, existing `Instability` and `VWSRSparsity` dispatch.
- Produces: `select_normalized_utopia(objective_vectors: Sequence[Sequence[float]]) -> int | None`; `run_nir1_epde(dataset: str, variant: str, seed: int, noise: float, *, overrides: dict[str, object]) -> dict[str, object]`; explicit research method names `nir1_criterion_only`, `nir1_regulator_only`, `nir1_combined`, `nir1_historical`.

- [ ] **Step 1: Write RED selector tests.** Handle empty front, constant objective columns, exact ties, nonfinite objectives; assert changing truth metadata changes no selected index; match current PIC compromise behavior only when `selector='production'`.
- [ ] **Step 2: Write RED EPDE adapter tests.** Exact `default` config must resolve to `wape/chi2/vwsr`; research variants remain distinct. On a mocked EPDE equation, custom objective and sparsity obey the active-mask layout and intercept rule. Multi-trajectory average is over trajectories, not the last one.
- [ ] **Step 2b: Write RED factorial-independence test.** On the same fixed equation, `criterion_only` keeps production coefficients/support while changing only the second Pareto objective, `regulator_only` changes selection policy but retains production `chi2` objective computation for an identical support, and `combined` enables both; defaults stay identical with absent `None` overrides.
- [ ] **Step 3: Run RED.** `python -m pytest .../test_selectors.py .../test_epde_adapter.py -q`. Expected: missing API.
- [ ] **Step 4: Implement fixed-front selectors and the narrowest explicit research hooks.** Add two `ObjectivesConfig` optional fields `research_objective_metric: str | None = None` and `research_regularizer_metric: str | None = None`; route them to `Instability.compute` and `PhysicsInformedLasso.fit` independently, falling back to the existing `instability_metric` when unset. New basis-free metric name `nir1_excess` must be registered in the **same** `_BASIS_FREE_METRICS` object consumed by objective and keep-rule dispatch, with validated `METRIC_MENUS`. Add `Nir1AdaptiveSparsity` as a dedicated operator class selected through `sparsity_cls='nir1_adaptive'`, but refactor only a minimal `VWSRSparsity` estimator factory hook if possible; do not duplicate full `VWSRSparsity.apply`. All additions are opt-in and live in the research branch. Enforce compatible Gram modes; never pretend old `vcoef` and current `chi2` share an uncertainty channel.
- [ ] **Step 5: Run GREEN and PIC full regression tests.** `python -m pytest projects/nir1_stability/tests/test_selectors.py projects/nir1_stability/tests/test_epde_adapter.py projects/pic/tests/test_runner_identity.py projects/pic/tests/test_metrics.py -q` passes. Run one default EPDE smoke on `ode`; record front and chosen model with same prior SHA baseline for controlled parity.
- [ ] **Step 6: Commit `feat(nir1): criterion and regularizer plugins for EPDE discovery`.**

## Slice D — Long-running, quality-preserving execution

### Task 9: Immutable planned-run ledger, bounded retries and resume

**Files:**
- Create: `projects/nir1_stability/nir1/campaign.py`, `projects/nir1_stability/nir1/records.py`
- Create: `projects/nir1_stability/tests/test_campaign_resume.py`
- Create: `projects/nir1_stability/manifests/README.md`

**Interfaces:**
- Consumes: `canonical_run_id`, `run_nir1_epde` and fixed-case audit runner.
- Produces: `plan_campaign(config: Mapping[str, object]) -> dict[str, object]`; `execute_shard(manifest_path: Path, shard_id: str, output_root: Path) -> int`; `resume_plan(ledger_path: Path, available_records: Sequence[Path]) -> dict[str, object]`; immutable `planned.json` listing every identity.

- [ ] **Step 1: Write RED interruption/retry tests.** Simulate process killed after two completed of five identities: saved rows are resumed once, missing three rerun with same seeds; malformed partial file not success; duplicate conflicting row raises `ValueError`. Infrastructure retry capped at 2, algorithm failure preserved.
- [ ] **Step 2: Write RED resource-control tests.** Reject identical `run_id` with different bytes; reject planned 7-hour single shard; when the 300-minute soft deadline is reached, retain completed local atomic files and explicitly mark remaining IDs incomplete; all statuses `ok/timeout/crash/unsupported/incomplete` remain enumerable and denominators reconcile.
- [ ] **Step 3: Run RED.** `python -m pytest projects/nir1_stability/tests/test_campaign_resume.py -q`. Expected: missing APIs.
- [ ] **Step 4: Implement one-run-at-a-time atomic writes (`tempfile` + `os.replace`), content hashes, append-only retry history, bounded shard scheduler and resume missing-only algorithm.** Do not log secrets; resume across runner restarts via downloaded matching artifacts.
- [ ] **Step 5: Run GREEN and two-round local campaign simulation.** All planned IDs exactly once, one flagged timeout not counted success; manifest summary totals match planned counts.
- [ ] **Step 6: Commit `feat(nir1): immutable resumable campaign ledger`.**

### Task 10: Exact-first acceleration and parity/fallback harness

**Files:**
- Create: `projects/nir1_stability/nir1/performance.py`, `projects/nir1_stability/nir1/cache.py`
- Create: `projects/nir1_stability/tests/test_exact_acceleration.py`
- Modify: `projects/nir1_stability/nir1/gram.py` and `nir1/regularizers.py`

**Interfaces:**
- Consumes: `ResearchDesign`, `GramBlocks`, `SparseFit`, canonical data and split identity.
- Produces: `make_cache_key(design: ResearchDesign, *, target_id: str, dtype: str, revision: str) -> str`; `fit_with_exact_cache(design: ResearchDesign, support: np.ndarray, cache: dict[str, object], *, l1: float, l2: float, weights: np.ndarray) -> SparseFit`; `compare_reference_fast(cases: Sequence[ResearchDesign], *, rtol: float = 1e-9, atol: float = 1e-11) -> list[dict[str, object]]`; performance CLI to generate `performance_regression.csv`.

- [ ] **Step 1: Capture E0 reference timing BEFORE changing the fast path.** Measure weighted Gram builds, solver iterations, stage time, RSS on 10⁴×20, 10⁵×50 (plus feasible combinations), record raw objective/support baseline.
- [ ] **Step 2: Write RED cache tests.** Same content and support subset hits; change any target, weights, token order or value, float dtype, normalization, code revision or derivative representation → miss. Reject stale/corrupt cache bytes via hash.
- [ ] **Step 3: Write RED equality/fallback tests.** Fast subset selection has identical support on seeded fixtures, `allclose` coefficients/objectives; ill-conditioned and near-threshold cases must fall back to SVD/reference and leave a recorded reason; `approximate=true` never appears in an exact result.
- [ ] **Step 4: Run RED.** `python -m pytest projects/nir1_stability/tests/test_exact_acceleration.py -q`. Expected: missing APIs.
- [ ] **Step 5: Implement stable Gram reuse by subset slicing, cache content key, optional factorization reuse, parallel shard scheduling without changing per-run RNG, and numeric KKT/condition-based fallback.** Do not enable subsampling, smaller evolution budgets or approximate covariance as speedup defaults.
- [ ] **Step 6: Run GREEN + paired equality and performance.** Tests pass; produce per-case support equality, coefficient/objective errors, fallback count, p50/p95 seconds/RSS and cache hits on all feasible sizes; any mismatch marked `quality_changed` and implementation switches that case to reference.
- [ ] **Step 7: Commit `perf(nir1): verified quality-preserving statistics reuse`.**

### Task 11: GitHub Actions smoke, sharded campaigns and artifact transfer

**Files:**
- Create: `.github/workflows/nir1-smoke.yml`, `.github/workflows/nir1-research.yml`
- Create: `projects/nir1_stability/tests/test_workflow_contract.py`
- Create: `projects/nir1_stability/manifests/launch/README.md` and small smoke launch manifest
- Modify: `projects/nir1_stability/README.md`

**Interfaces:**
- Consumes: `plan_campaign`, `execute_shard`, test suite, PIC CPU dependency setup.
- Produces: branch-scoped `push` smoke on code paths and research workflow triggered **only** by approved launch-manifest changes; Actions artifacts `nir1-ledger-<hash>` and `nir1-results-<stage>-<shard>`.

- [ ] **Step 1: Write RED YAML contract tests.** Parse each workflow with PyYAML (`on` key must be handled safely); assert trigger branch is exclusively research branch; research trigger path only `projects/nir1_stability/manifests/launch/**`; permissions read-only; job timeout ≤360; upload artifacts `if: always()`; no secrets or uploads to upstream.
- [ ] **Step 2: Run RED.** `python -m pytest projects/nir1_stability/tests/test_workflow_contract.py -q`. Expected: workflow files absent.
- [ ] **Step 3: Implement workflows using checkout, `astral-sh/setup-uv`, `uv sync --project projects/pic --extra cpu --locked`, pinned BLAS thread count, explicit shard matrix, artifact uploads and traceable code revision.** Set `timeout-minutes: 340`, runner's internal soft deadline to 300 minutes, allow at least 30 minutes for upload; never assume local files survived an unexpectedly killed VM. Workflow shall not assume manual `workflow_dispatch` from research branch; the default branch is untouched.
- [ ] **Step 4: Run GREEN locally.** Tests pass, YAML parses; validate GitHub workflow status after push using read-only connector. If remote rejects workflow edit/trigger, stop at permission boundary and hand the user a precise manual action; do not bypass with credentials.
- [ ] **Step 5: Commit `ci(nir1): isolated smoke and sharded research Actions`, push **research branch only**, observe first actual Actions smoke run.** Record run ID, commit SHA, logs, exit state and artifact checksum.

## Slice E — Statistical evidence and presentation

### Task 12: Honest paired statistics and reviewer-ready report generator

**Files:**
- Create: `projects/nir1_stability/nir1/metrics.py`, `projects/nir1_stability/nir1/reporting.py`
- Create: `projects/nir1_stability/tests/test_metrics_reporting.py`
- Create: `projects/nir1_stability/METHODS.md`, `projects/nir1_stability/REVIEW.md`, `projects/nir1_stability/RESULTS.md`

**Interfaces:**
- Consumes: complete ledger and per-run records, PIC canonical truth parser, fixed-front scores, CPU time/memory.
- Produces: `summarize_outcomes(ledger: Mapping[str, object], records: Sequence[Mapping[str, object]]) -> dict[str, object]`; `paired_effect_ci(a: Sequence[bool], b: Sequence[bool], groups: Sequence[str], *, n_boot: int = 2000, seed: int = 0) -> dict[str, float]`; `render_review(summary: Mapping[str, object], output_dir: Path) -> list[Path]`.

- [ ] **Step 1: Write RED denominator tests.** 10 planned = 5 ok, 2 crash, 1 timeout, 1 unsupported, 1 incomplete: status totals must sum to 10; supported planned denominator = 9, with crash/timeout/incomplete counted not-success; standalone conditional on `ok` separately labeled. Truth on front != selected success; no `unsupported` as algorithm failure.
- [ ] **Step 2: Write RED inferential and reporting tests.** Paired identical outcomes produce Δ=0; cluster bootstrap resamples system IDs, not points; empty/partial data yields `PROVISIONAL` and `NOT RUN`; changing method truth labels does not retroactively alter selected model; figure builder works headless.
- [ ] **Step 3: Run RED.** `python -m pytest projects/nir1_stability/tests/test_metrics_reporting.py -q`. Expected: missing APIs.
- [ ] **Step 4: Implement Wilson, McNemar, paired bootstrap, Holm for preregistered contrasts, per-system summaries and 5–8 scientific figures; render Markdown tables/figures strictly from records with source run IDs.** Include historical `58/76` as *literature-like archived finding*, never current rerun success.
- [ ] **Step 5: Run GREEN plus render an intentionally partial test campaign.** Test suite passes; report explicitly says incomplete, with status ledger and no fabricated conclusion.
- [ ] **Step 6: Commit `docs(nir1): paired evidence and review-ready reporting`.**

### Task 13: Run S0, inspect S1 feasibility, freeze confirmation protocol

**Files:**
- Create: `projects/nir1_stability/manifests/launch/s0-2026-10-10.yaml`
- Create: `projects/nir1_stability/manifests/s1-frozen.yaml`
- Generate: `projects/nir1_stability/reports/s0/` and pilot ledger/artifacts

**Interfaces:**
- Consumes: Tasks 1–12 full smoke stack and Actions workflow artifact fetch.
- Produces: S0 truth/decoy ranking, identifiability failure maps, 6-system S1 frozen candidate methods and measured shard wall-times, no test-set tuning.

- [ ] **Step 1: Verify `nir1-smoke` GitHub Actions GREEN and artifacts accessible.** Expected: first real GitHub run ID and tests; a green badge without artifacts does not satisfy evidence gate.
- [ ] **Step 2: Commit a new immutable S0 launch file and observe Actions runs.** Run T1–T8, 14-system compatible cases on available data, per-method seed grid; preserve all statuses and artifacts. Expected: complete or visibly partial manifest with counts and SHA.
- [ ] **Step 3: Analyze negative controls.** Specifically demand per-column scaling, correlation 0.999, Allen–Cahn low diffusion, duplicate token and variable-coefficient condition; unexpected behavior is a documented defect/negative result, not deleted.
- [ ] **Step 4: Run S1 pilot after S0 gate.** Six supported representative systems × 5 paired seeds × baseline + criterion-only + regulator-only + combined + historical + matched PySINDy where supported; produce per-stage profiler and failure ledger.
- [ ] **Step 5: Freeze finalists and run budget.** Set S2 methods/splits/seeds before testing heldout; apply planned finite resource cap from actual p95/maximum pilot duration and available RAM. If S1 cannot fit into ≤360-minute shards, optimize E1 and rerun parity checks first.
- [ ] **Step 6: Commit `research(nir1): reproducible S0 and S1 audit evidence`.** Do not commit huge binary datasets; commit report index, provenance and small figures, preserve raw data in Actions artifacts with manifest/hash.

### Task 14: Heldout S2/S3 experiments, final audit and concise handoff

**Files:**
- Create: `projects/nir1_stability/manifests/launch/s2-confirmation.yaml`, `projects/nir1_stability/manifests/launch/s3-transfer.yaml`
- Generate: `projects/nir1_stability/reports/final/`, final `REVIEW.md`, `RESULTS.md`, `METHODS.md`, `README.md` updates
- Create: `projects/nir1_stability/Кратко_для_ревью.md` (or ASCII basename plus Russian text when path stability demands it)

**Interfaces:**
- Consumes: S1 frozen finalists and resource estimate, strict trained-vs-heldout split, reporter.
- Produces: independently scored S2 paired runs (8–12 systems ×20 seeds×3–4 finalists where feasible), S3 diagnostic strata, complete status reconciliation, plots/tables, run links, concise English/Russian research handoff.

- [ ] **Step 1: Check S2 launch manifest against frozen config.** No changed method selection, selector thresholds or test systems after first heldout score; test manifest hash equality with S1 decision output.
- [ ] **Step 2: Launch sharded S2 by immutable manifest, monitor with non-destructive read-only Actions status, and resume only missing eligible IDs after complete checkpoint preservation.** Expected: every planned ID either `ok`, `timeout`, `crash`, `unsupported`, or explicitly `incomplete`; no unlabelled omissions.
- [ ] **Step 3: Run S3 on independent initial conditions, wider pool, variable coefficients, KS/NS where the measured RAM budget permits, and bounded 0.5–2% diagnostic noise only as a separate stratum.** Clearly distinguish unsupported/too-large tasks and deferred experiments.
- [ ] **Step 4: Generate all reports and audit their claims against raw records.** Include exact selected recovery, front discovery, selection gap, false-deletion rate, CI, paired effect, time/RAM quality–cost curves, failure counterexamples, and absent-data caveats. A method not significantly better is not declared superior.
- [ ] **Step 5: Verify no upstream mutations and run final tests.** `git remote -v`, `git branch --show-current`, `git status --short`, `python -m pytest projects/nir1_stability/tests projects/pic/tests -q`; Actions final linked SHA and artifact checksums match review report.
- [ ] **Step 6: Commit `research(nir1): frozen confirmation results and review packet` on research branch only.** Do not open or merge upstream PR. Review package contains experiment tables, links, methods and recommended next scientific decisions.

---

## Fixed decision gates and completion language

1. **S0 gate:** failing scale/permutation invariance or silent collinearity errors must be fixed before any claim of criterion improvement.
2. **S1 gate:** split criterion-only vs regularizer-only gains; if only front improves, do not claim final selection improved. If runtime exceeds budget, E1 optimization parity comes before S2.
3. **S2 gate:** record the point estimate and uncertainty even if under +10 percentage-point target; no parameter retuning on heldout tasks.
4. **Approximation gate:** E2 speedups receive an `approximate=true` marker, separate metric table, and no 'without quality loss' claim absent comprehensive paired parity.
5. **End-of-session claim:** “implemented” requires code SHA and test evidence; “experiments completed” additionally requires Action run IDs, complete/partial manifest counts and result artifacts; “improved EPDE” additionally requires a predeclared statistical comparison and documented negative cases.

## Coverage check

| Spec requirement | Implementing tasks |
|---|---|
| Isolation, provenance and reproducibility | 1, 3, 9, 11 |
| T1–T8 and hypothesis-driven diagnostics | 2–4, 7, 13 |
| Existing objectives, historical comparisons, adaptive regulation | 4–8 |
| Fixed list, frozen Pareto and full EPDE decomposition | 7, 8, 12–14 |
| Identifiability, variable coefficient and abstention diagnostics | 3–6, 12–14 |
| Quality-preserving exact optimization, no silent E2 | 2, 9–11, 13–14 |
| Long Actions, chunking, resume and truthful failure counts | 9, 11–14 |
| Paired CI, heldout performance and reviewer handoff | 12–14 |

## Execution handoff

The user selected GitHub Actions execution in a private research **branch of their public fork**. After approving this implementation plan, use `superpowers:using-git-worktrees` and `superpowers:test-driven-development`, then execute tasks serially with `superpowers:executing-plans` or task-by-task reviews where available. Persist exact commit SHAs, one ledger line per completed task, and a final independent whole-branch review. Stop at any explicit permission barrier; never expand authority to upstream ITMO or transfer credentials.
