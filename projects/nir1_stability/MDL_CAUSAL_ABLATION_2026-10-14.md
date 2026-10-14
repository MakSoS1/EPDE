# H10 length-penalty causal ablation — frozen before new runs (2026-10-14)

**Scope:** Only `MakSoS1/EPDE` branch `research/nir1-mdl-factorial-2026-10-14`; never upstream, `main`, or `nir1`. These tests respond to the scientific objection that model-length regularization is not independently helpful in equation discovery.

## Primary causal contrast

Compare `mdl_min_repair` versus `no_mdl_min_repair` with exactly the same time-target prior, RPS, final argmin selector, repair logic, VWSR, token libraries, data and evolutionary budgets. Only `NIR1_DYNAMIC_MDL` changes between the arms.

Secondary factorial arms:
- `native`: original PIC configuration, no time prior, standard two-objective Pareto compromise
- `anchor_utopia`: H9 time target, native final compromise
- `anchor_min`: H9 time target, min-first-objective final selection, no repair, no MDL
- `mdl_utopia`: H9 + H10, native compromise, no repair
- `mdl_min`: H9 + H10 + H11, no repair
- `mdl_min_repair`: H9 + H10 + H11 + H12
- `no_mdl_min_repair`: H9 + same final selector + same H12 repair, WITHOUT MDL.

Each variant gets its own fresh subprocess. No data labels or test truth are used to choose target, fitness, coefficients, front, or selected index. Structural truth is read *after* selection to score the outcome.

## Frozen execution plan

- **36** previously seen PDE/ODE physical systems, frozen before launching this factorial. They are **not independent new heldout systems**.
- **7** arms per system, **3** paired optimizer seeds 0, 1, 2.
- `36 * 7 * 3 = 756` full EPDE evolutions; no smoke run enters the full results.
- Data seed 0, single trajectory as available in the existing benchmark, noise 0, unchanged native per-dataset token pool/preprocessing and search budgets (normally 16×5; Lotka–Volterra uses 32×5 for all arms). Exact configuration in each JSON is authoritative.
- E1 off: `NIR1_DISABLE_GRAM_CACHE=1`, `NIR1_CHI2_PREFIX_CACHE=0`.
- Code runs in `ubuntu-24.04`, Python 3.12, pinned `uv.lock`; native CLI runner is `projects.nir1_stability.nir1.mdl_factorial_runner`.
- Three trigger-on-commit workflows, one per seed; each has a 252-job `dataset×arm` matrix, under GitHub's 256 matrix combination cap. `max-parallel=4` per workflow; `timeout-minutes=105` per search. Smoke preflight checks the two newly introduced arms on `ode` before releasing the full matrix.
- Each full run emits an artifact with `native-full.json`, `source-sha.txt`, and protocol text. Upload even when search crashes. A failed job is **counted as a failure**, not excluded. No automatic reruns or silent retries. Missing jobs are treated as incomplete until explicitly reconciled.

## Predeclared evaluation

1. **Primary:** paired difference of selected structural exact recovery, `mdl_min_repair - no_mdl_min_repair`, across the 36 systems; report per-system 3-seed results, clusters on **systems**, not 108 nominally independent observations.
2. **Secondary:** truth-on-Pareto, Hamming, crash/timeout, runtime, MDL effects under `mdl_min` vs `anchor_min`, repair effect `mdl_min_repair` vs `mdl_min`, final selector effect `mdl_min` vs `mdl_utopia`.
3. **Required failure strata:** 3rd/4th spatial derivative, variable-coefficient, damped wave, traveling-wave/low rank, nonlinear ODE. No excluding hard cases post hoc.
4. Independent two-sided sign-flip or clustered bootstrap with caution regarding just 36 development-used systems; report seed-by-system table, not just aggregate accuracy. For multiple confirmatory contrasts use Holm correction.
5. `no_mdl_min_repair` and `mdl_min_repair` comparison specifically **isolates MDL within the implemented environment**, but no claim of external generalization; all 36 systems are development-contaminated from previous studies.
6. If a variant changes candidate support or fronts due to MDL, that is an intentional part of the causal effect. Offline front replay alone cannot replace an end-to-end test.

## Frozen datasets (in order)

```text
ac burgers duffing lv ode vdp
nir1_decay nir1_harmonic nir1_logistic nir1_quadratic nir1_cubic
nir1_advection nir1_heat nir1_reactiondiff nir1_wave nir1_advdiff
nir1_h11_gaussian_heat nir1_h11_advection_decay nir1_h11_viscous_burgers
nir1_h11_klein_gordon nir1_h11_biharmonic nir1_h11_airy
nir1_h11_shear nir1_h11_logistic_field nir1_h11_gauss_reactdiff
nir1_h14_damped_wave nir1_h14_cole_hopf nir1_h14_kpp_front
nir1_h14_linear_ks nir1_h14_beam nir1_h14_airy_transport
nir1_h14_cubic_saturation nir1_h14_driven_advection
nir1_h14_variable_diffusivity nir1_h14_liouville_growth
nir1_h14_hyperdiffusion_growth
```

## What this does NOT test

- Original unmodified ITMO upstream is not a seventh matched arm (the `native` arm is PIC baseline in user's fork).
- No genuinely new physical systems (separate future external validation).
- No noisy/spectral/polynomial derivative methods, no new initial conditions, no cross-framework comparison.
- This is NOT proof that `log(N_time)/N_time` is statistically optimal: MDL and its order weight stay frozen here.

**Do not use results produced by this protocol to retune H10 and simultaneously call them a fresh holdout confirmation.**
