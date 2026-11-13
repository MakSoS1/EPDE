# Reproducible multisample benchmarks

The [Russian implementation and research report](REPORT_RU.md) explains the current branch, the joint-fit mathematics, repaired sample handling, measured results, limitations, and a complete runnable example. The [verification protocol](VERIFICATION.md) records the baseline and final test results, including the five pre-existing solver digest failures and the repository-wide collection blockers.

Shared-coefficient fitting is enabled with `multisample_mode='joint'` and `multisample_weighting='samples'` on `EpdeSearch` or `EpdeMultisample`, using `system_rps_mode='plain'` or `'assignment'`. The default remains `'mean'`. This mode preserves independent trajectory preprocessing and combines evaluated rows only for fitting and proposal screening; it does not silently redefine the existing discrepancy or instability metrics.

`benchmark.py` evaluates the common-coefficient multisample contract on independent trajectories and fields. It uses EPDE's real fitting operators and public trajectory/domain construction API. Analytic fixture checks, fitting supplied terms, selection among supplied terms, and evolutionary discovery are reported separately.

## Run

Run from the repository root with the Python environment in which EPDE and its dependencies are installed:

```bash
python projects/multisample/benchmark.py --check-data
python projects/multisample/benchmark.py --suite all --list
python projects/multisample/benchmark.py --suite smoke --output projects/multisample/results/smoke.json
python projects/multisample/benchmark.py --suite fixed --workers 3 --output projects/multisample/results/fixed.json
python projects/multisample/benchmark.py --suite discovery --workers 3 --output projects/multisample/results/discovery.json
python projects/multisample/benchmark.py --suite compatibility --workers 3 --output projects/multisample/results/compatibility.json
```

The default smoke suite contains four fixed-structure jobs. The complete default matrix contains **98 jobs**: 32 fixed-structure jobs, 60 evolutionary discovery jobs, and six compatibility jobs. Discovery uses seeds `0,1,2,3,4`, a requested population of eight, and four epochs. EPDE may expand its initial population according to its own diversity rules; the hard timeout remains the resource bound.

Every job runs in a new subprocess because EPDE configuration and caches use process globals. BLAS, OpenMP, NumExpr, and Torch use one CPU thread, and CUDA is disabled. At most three subprocesses run concurrently. Default hard limits are **45 seconds for fixed/assignment jobs** and **90 seconds for discovery jobs**, including worker startup. Change them with `--fixed-timeout` and `--discovery-timeout`.

The diagnostic runner sets `runtime.free_tensor_cache_after_fit=False` so that it can evaluate training residuals after `fit()` returns. EPDE's normal cache release is unchanged. Fixed-structure weak runs explicitly prepare the weak rows, and every job records and checks the actual evaluation form.

Filters can select a small diagnostic experiment:

```bash
python projects/multisample/benchmark.py --suite fixed --scenarios reaction_diffusion --derivatives analytic --output projects/multisample/results/pde_identifiability.json
python projects/multisample/benchmark.py --suite fixed --scenarios density --output projects/multisample/results/density.json
python projects/multisample/benchmark.py --suite discovery --scenarios logistic --derivatives analytic --seeds 0,1,2,3,4 --workers 3 --output projects/multisample/results/logistic_discovery.json
python projects/multisample/benchmark.py --suite discovery --scenarios coupled --modes joint --noise 0.01 --seeds 0 --output projects/multisample/results/coupled_noisy.json
```

`--list` prints the selected matrix without importing EPDE or fitting. `--check-data` validates the analytic generators and rank argument only; it is explicitly not an EPDE benchmark result. No conclusions about improvement are implied by the existence of this runner.

## What each stage measures

| Stage | EPDE path | Information supplied to EPDE |
|---|---|---|
| `fixed` | `plain_fit._versions`, coefficient calculator, discrepancy and instability fitness | Correct nonconstant terms and canonical target; the affine case retains production choice of the intercept |
| `assignment` | `SoEqRightPartSelector.apply` in assignment mode | A candidate term set; EPDE selects targets/supports and coefficients |
| `discovery` | `EpdeSearch.fit`, plain mode, NSGA-II | Data and a bounded token library; initial population contains no inserted true equations |

The fixed and assignment stages call internal operators intentionally, following the same construction as the repository's operator integration tests. They do not replace EPDE with a NumPy least-squares fit. They retain production rank, instability, amplification, and candidate acceptance checks. A mathematically identifiable stacked library can therefore still produce `no_candidate` if another production rule refuses it.

The `mean` mode is the compatibility baseline. The `joint` mode requests shared-coefficient fitting. Most comparisons use `multisample_weighting='samples'`; the density control also uses `'points'`. All strong-form experiments supply a uniform domain weight to separate sample aggregation from spatial weighting. Each sample owns its grid, observations, derivatives, domain ID, and trajectory ID. Held-out records are never registered in EPDE's training pool.

## Analytic scenarios

| Scenario | Governing equation | Purpose |
|---|---|---|
| `affine` | `u_t = 0.4 - 0.7 u` | Shared intercept and unequal lengths |
| `logistic` | `u_t = 1.2 u - 0.6 u²` | Nonlinear feature and different amplitudes/initial conditions |
| `coupled` | `u_t = -0.2 u + v`, `v_t = -u - 0.2 v` | Two variables on two independent trajectories, cross-variable terms |
| `reaction_diffusion` | `u_t = 0.1 u_xx - 0.2 u` | Different spatial grids and complementary identifiability |
| `contradictory` | Separate samples obey `u_t = -0.4u` and `u_t = -1.0u` | Negative control: a common coefficient cannot fit both exactly |
| `density` | Separate samples obey `u_tt = -u` and `u_tt = -4u` | Exact sample-weight versus point-weight response to grid refinement |

ODE samples use distinct time origins and unequal lengths. The PDE training grids have shapes `(31, 41)` and `(23, 53)` in `(time, space)` order and disjoint spatial intervals. The clean PDE holdout uses a new grid, a disjoint spatial interval, and a mixture of Fourier modes that was not used for training.

For a PDE training sample,

```text
u_s(x,t) = A_s exp[(-0.2 - 0.1 k_s²)(t-t_start)] sin(k_s x),
k_1 = 1, k_2 = 2.
```

Each individual design `[u, u_xx]` has rank one because `u_xx = -k_s²u`. The stacked design has rank two. The report stores individual and stacked singular values for both the exact fixture and the EPDE-evaluated fitting rows. This argument establishes information in the data; accepted candidate recovery and evolutionary discovery are separate measured outcomes.

The density control samples a complete period without its duplicate endpoint and uses zero boundary exclusion. Both signals have exactly the same sampled mean square, `1/2`, at the chosen resolutions. Refining the second sample from 64 to 256 points leaves the correct sample-balanced fitted coefficient at `-2.5`; point weighting changes it from `-2.5` to `-3.4`. This control deliberately contains conflicting coefficients so that weighting mistakes cannot hide behind an exact common-law fit. Its summary checks these analytic expectations with a fixed absolute tolerance of `1e-7`.

## Derivatives and noise

`analytic` passes exact derivatives of the clean analytic solutions into each trajectory. These runs isolate the fitting and search implementation from differentiation error.

`fd` lets EPDE independently differentiate each sample using its FD preprocessor. Fixed-structure tests include clean FD and observation noise levels `0.01` and `0.05`; discovery includes clean analytic derivatives and `0.01` noisy FD. Gaussian noise is independent per sample and variable, with standard deviation `noise * std(clean field)`. No clean derivative oracle is supplied in noisy FD runs. This is intentionally a direct finite-difference stress test; the runner does not tune a denoiser to the holdout.

The compatibility suite also tries two clean weak-form fixed-structure jobs. These only check whether the existing weak-row path works with the multisample interface and recovers usable physical coefficients. They do not establish a full weak-form noise benchmark or modify the weak-form implementation.

## Holdout and model selection

Whole clean trajectories/fields with new initial conditions are held out. The selected model minimizes the mean training fitness among the returned candidates. The report keeps every returned model and never selects by coefficient error, support recovery, or holdout error.

Canonicalization rewrites each equation against its scenario's first time derivative, or the second time derivative for the density control. A model lacking that linear derivative receives an explicit unavailable reason for canonical coefficient comparison. Active support uses the declared threshold `1e-8 * max(1, max(abs(coefficients)))`; this threshold is fixed before running any experiment. Extra terms contribute to coefficient error even when the true coefficient is zero.

The report includes:

- EPDE training residuals for every sample, on the actual strong/weak fitting rows.
- Residuals on the underlying clean training fields and on clean held-out fields, evaluated independently from the serialized physical terms.
- Relative coefficient error and support match when a common true coefficient vector exists.
- Forward integration error for explicit first-order ODE models whose right sides contain only state variables. Unsupported or failed integration has a reason and no invented score.
- Wall time, in-worker time, seeds, library versions, source revision, working-tree diff hash, resolved configuration, equations, rank diagnostics, and per-job logs.

This matrix uses the same search budget for mean and joint modes. It does not establish a universal advantage, and it does not include a matched-total-points single-trajectory campaign. More information can help identify a shared law; extra incompatible or poorly differentiated data can hurt it.

## Files and result schema

For `--output results/run.json`, the runner writes:

- `results/run.json`: complete report, atomically checkpointed after each finished job.
- `results/run.csv`: one selected-model equation per row, including errors/timeouts that have no equation.
- `results/run_runs/*.json`: individual job outcomes.
- `results/run_runs/*.log`: unfiltered worker stdout/stderr for troubleshooting.

Schema version 1 has top-level `metadata`, `matrix`, `results`, and `summary`. Each result carries its `spec`, status, timing, training/holdout sample descriptions, fixture evidence, fit design diagnostics when applicable, and all returned models. Successful results identify `selected_model_id` and the training-only selection rule.

Statuses distinguish `ok`, `no_candidate`, `error`, and `timeout`. A refused candidate is not silently reported as a successful recovery. A timeout is not discarded from the denominator. JSON is strict: nonfinite values become `null`, and their paths/reasons are retained under `unavailable_values`; unavailable metrics also have contextual reason fields. Existing run directories should not be shared by concurrently launched campaigns.

`--resume` can continue an interrupted campaign with exactly the same matrix. It archives the earlier report, retains completed non-error outcomes (including timeouts and `no_candidate`), and retries errors and unfinished jobs. Earlier error logs are archived rather than overwritten. The report records which result IDs were retained and the runner fingerprint that produced them.

## Measured campaign: 2026-11-13

The recorded campaign is in `results/2026-11-13/benchmark.json`, with selected-equation rows in `benchmark.csv` and complete individual outcomes/logs in `benchmark_runs/`. It contains **98 final outcomes: 97 returned a candidate and one returned `no_candidate`; no final errors or timeouts occurred**. Returning a candidate is a completion status, not a claim of correct equation discovery.

The one refusal is the mean-mode fixed-structure analytic PDE, whose feature columns are rank deficient on each individual sample. Joint mode accepts the same supplied structure and recovers `u_t = -0.2 u + 0.1 u_xx` with relative coefficient error `5.00e-16` and held-out mixed-mode residual `7.76e-16`.

### Evolutionary discovery

Each row below represents five seeds, population request eight, four epochs, and training-only model selection. Coefficient error is the median of the maximum relative equation coefficient error in each returned system; the holdout column is the median mean relative residual across equations. Noisy cases use 1% observation noise and EPDE finite differences. These are compact candidate libraries, not a broad claim about arbitrary nonlinear PDE search.

| Problem and derivatives | Mode | Exact support | Coefficient error | Holdout residual | ODE rollout error |
|---|---|---:|---:|---:|---:|
| Logistic, analytic | mean | 5/5 | 6.62e-16 | 7.36e-15 | 1.08e-9 |
| Logistic, analytic | joint | 5/5 | 1.85e-16 | 3.60e-15 | 1.08e-9 |
| Coupled ODE, analytic | mean | 5/5 | 5.44e-17 | 5.50e-17 | 4.37e-9 |
| Coupled ODE, analytic | joint | 5/5 | 2.33e-16 | 2.58e-16 | 4.37e-9 |
| Reaction–diffusion, analytic | mean | 0/5 | 1.204 | 0.5525 | Not evaluated |
| Reaction–diffusion, analytic | joint | 5/5 | 1.86e-16 | 2.64e-16 | Not evaluated |
| Logistic, noisy FD | mean | 5/5 | 0.001567 | 0.01823 | 0.001713 |
| Logistic, noisy FD | joint | 0/5 | 0.003846 | 0.003392 | 0.0003362 |
| Coupled ODE, noisy FD | mean | 0/5 | 0.003589 | 0.002666 | 0.004436 |
| Coupled ODE, noisy FD | joint | 0/5 | 0.002672 | 0.002436 | 0.004175 |
| Reaction–diffusion, noisy FD | mean | 0/5 | 1.204 | 0.5525 | Not evaluated |
| Reaction–diffusion, noisy FD | joint | 0/5 | 0.1289 | 0.07936 | Not evaluated |

The clean PDE gives the clearest demonstrated benefit: complementary samples supply information that separate per-sample fitting loses. The clean ODE tests recover the same law under either mode. Tiny differences at floating-point precision are not meaningful improvements.

The noisy logistic case shows a real tradeoff. Joint mode retains a small extra intercept in every seed, so its strict support score is worse and its median coefficient error is larger. Nevertheless, its held-out residual and forward trajectory error are smaller in all five paired seeds. The noisy coupled system has extra terms in both modes, and joint mode is not better for every seed. Thus the evidence supports joint coefficient estimation as a distinct objective, not a universal guarantee of better model selection.

For direct inspection of variability, the noisy discovery holdout residuals are:

| Problem / mode | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|---|---:|---:|---:|---:|---:|
| Logistic / mean | 0.0182274 | 0.0070610 | 0.0049477 | 0.0259644 | 0.0381435 |
| Logistic / joint | 0.0033922 | 0.0001478 | 0.0021243 | 0.0082155 | 0.0057249 |
| Coupled / mean | 0.0027743 | 0.0017666 | 0.0026658 | 0.0020942 | 0.0028471 |
| Coupled / joint | 0.0024359 | 0.0019260 | 0.0025736 | 0.0017265 | 0.0036316 |
| Reaction–diffusion / mean | 0.5517697 | 0.5520905 | 0.5524607 | 0.5526002 | 0.5526495 |
| Reaction–diffusion / joint | 0.0807709 | 0.0793553 | 0.0759359 | 0.0814957 | 0.0732473 |

All clean analytic logistic/coupled seeds recover exact support in both modes; all clean analytic PDE seeds recover it only in joint mode. For noisy discovery, logistic mean succeeds in all five strict support checks; all other noisy rows fail that strict check. The raw report retains each equation, coefficient, objective, and rollout result, not only these summaries.

### Weighting, incompatibility, and compatibility controls

The density control matches its exact prediction: refining the second grid changes the sample-balanced coefficient by **0.0**, from `-2.5` to `-2.5`, while point weighting changes it to `-3.400000000000002`. The conflicting first-order laws also retain nonzero per-sample residuals; the largest clean training residual is `0.75` in mean mode and approximately `0.4537` in joint mode. The intermediate-rate holdout of this negative control is descriptive only: there is no common true coefficient vector against which to rank a recovered law.

Both weak-form fixed-structure compatibility jobs execute actual weak rows and recover the supplied structure: logistic coefficient error `4.31e-14`, PDE coefficient error `7.73e-10`. This validates those two clean interface combinations only.

Assignment selection recovers logistic in both modes. On the PDE, its single seeded selection chooses a different Pareto variant: mean mode returns a spatial-only relation, and joint mode returns a relation equivalent to a diffusion-only PDE with coefficient error approximately `0.9888`. These are successful operator executions, not successful recovery of the complete reaction–diffusion law. Assignment integration must not be described as structurally correct on the strength of the plain-mode result.

The verified final jobs total approximately **746.4 worker seconds**, including worker imports; the slowest job takes **54.5 seconds**. Some noisy coupled searches expand the initial population and repeatedly exhaust unique-offspring proposals. Those messages and timings remain in the logs. The resumed campaign's wall interval is approximately four minutes with three workers; the earlier preflight and initial fixed jobs are separate from that interval.

### Execution audit and limits

An initial reporting attempt completed searches but then tried to evaluate equations after EPDE's expected tensor-cache release. Its 14 discovery reporting errors are preserved in `benchmark_attempt_1.json` and the corresponding `.attempt_1.log` files. The diagnostic configuration was corrected to retain tensors, a discovery smoke run verified it, and the campaign resumed. Its already verified 32 fixed jobs were retained. These were runner postprocessing errors, not failed scientific searches, and they are not hidden or relabelled as algorithmic outcomes.

The source revision, dirty working-tree information, runner fingerprint, versions, seed, resolved configuration, and sample/grid descriptions accompany the results. The original campaign's `working_tree_diff_sha256` covers git-tracked changes only and does not uniquely fingerprint untracked implementation files. Its provenance is preserved as recorded. Future runs additionally hash the actual contents of every EPDE Python/JSON file and this runner, including untracked files. `post_review_source_manifest.json` fingerprints the later reviewed source separately; it must not be mistaken for a retrospective fingerprint of the earlier execution.

The 2026-11-13 source constancy-filter follow-up is checked separately in `constancy_validation.json`; every expected field and derivative token remains in the prepared libraries for all 16 distinct strong preprocessing configurations. The available logistic square also remains nonconstant. No affected fitting job was identified, so the campaign was not repeated after that change.

The finite-difference PDE runs use a boundary exclusion of two points on each axis, while the preprocessor reports a four-point edge reach for the second spatial derivative. Consequently some retained rows use one-sided stencil treatment. Both modes see the same rows, and the warnings are preserved in the logs; absolute FD error includes this boundary choice as well as noise and differentiation error. The exact-derivative rank and recovery results do not depend on that FD edge treatment.

Five seeds, a compact search space, one clean holdout per scenario, and a small search budget provide bounded regression evidence. They do not establish performance across all dynamics or noise distributions. The fixed noisy experiments use seed zero only. No full PDE forward solve, weak-form noise campaign, independent equal-total-points single-trajectory baseline, or hyperparameter sweep is claimed.
