# НИР-1 — H11: prospectively test fitness-consistent Minimum Dynamic Description

**Decision frozen 11.10.2026, before running ANY new H11 physical laws.**
**Code:** `nir1/causal_mdl_selector.py`, `nir1/epde_adapter.py`, `epde/operators/common/causal_mdl.py`.
**Source fork only:** `MakSoS1/EPDE:nir1`. E1 optimizations are frozen/off.

## Why the original NIR1 methods failed

The PIC EPDE search sometimes prefers a low-residual *spatial identity*
(`u_x u_xx = u u_xxx`), which contains no time dynamics. H9 forces a
**pure bare time derivative** as implicit target and uses the same original
VWSR + chi2, 16 individuals, 5 epochs and training data. H9 did not
substantially raise structural success (2/8 same as baseline).

H10 additionally penalizes nonzero RHS complexity and derivatives above
second order by `log(N_time) / N_time`, which is a time-level heuristic,
**not an exactly calibrated BIC for autocorrelated PDE data**. It adds the
cost to the **actual evolutionary fitness**, not just after search.

## Development evidence — not independent validation

Full EPDE historical development experiment
[38132787691](https://github.com/MakSoS1/EPDE/actions/runs/38132787691),
frozen search SHA `9b4f4675fe68a31bc188a987fc31f7bed23dcc18`,
`nir1_heat/nir1_advdiff/nir1_reactiondiff/nir1_wave`, two optimizer seeds.

| Method | Correct law somewhere on Pareto front | Correct law chosen by native PIC |
|---|---:|---:|
| default | 2/8 | 2/8 |
| H9 dynamic temporal target | 2/8 | 2/8 |
| H10 temporal target + dynamic MDL | **8/8** | 2/8 |

**Post-hoc inspected result:** in each H10 Pareto front the true equation had
the lowest **first objective (the actual evolved MDL objective)**, yielding
8/8 with a label-blind argmin-of-first-axis decision. However this decision
rule was selected AFTER reviewing development results. These 8/8 are NOT a
confirmatory scientific improvement.

## Locked H11 algorithm before unseen outcomes

1. Run native H10 full EPDE with no changes: original PIC library,
   preprocessing, single 81×96 trajectory, `population_size=16`,
   `training_epochs=5`, no Gram/E1 optimizations.
2. Candidate selection is
   `argmin_i objectives[i][0]`, for complete finite Pareto vectors; ties
   preserve recorded order. This is `select_min_causal_description`.
   No truth, known coefficients or oracle derivative order involved.
3. H11 is the `nir1_dynamic_mdl_select` opt-in variant, restoring stateful
   environment flags on exit and preserving native PIC score separately as
   `metrics_pic_original`.
4. Only after selection, score exact structure against the isolated benchmark
   truth list using `canonical_tokens`/Hamming. All statuses count, with
   failures, timeouts and unsupported runs visible.

### Nine new distinct physical laws, created before outcomes

- `nir1_h11_gaussian_heat`: non-Fourier Gaussian heat.
- `nir1_h11_advection_decay`: three-mode advection with exponential decay.
- `nir1_h11_viscous_burgers`: genuinely nonlinear viscous traveling shock.
- `nir1_h11_klein_gordon`: second-order temporal wave + nonzero mass.
- `nir1_h11_biharmonic`: **true fourth-order spatial derivative**, a
  deliberate challenge to H10 overregularization.
- `nir1_h11_airy`: **true third-order spatial derivative**, another
  deliberate high-order counterexample.
- `nir1_h11_shear`: variable-coefficient shear advection.
- `nir1_h11_logistic_field`: reaction law in spatially varying field.
- `nir1_h11_gauss_reactdiff`: non-Fourier Gaussian reaction–diffusion.

These laws may not all be equally difficult and are all synthetic and
correlated by physics class. Numerical PDE residuals are validated before
any full EPDE run. No selection of only easy outcomes.

## Equal data/evolution comparison

Per law and per paired seed 0/1, run:
- `default`, unchanged PIC (regularization, evolution, library and data);
- `nir1_dynamic_mdl_select`, H10 evolutionary fitness and H11 final
  selection, **same 16×5 budget and same trajectory**.

Total 9×2×2 = 36 full searches for a scientific *validation* of a new idea,
not to improve the algorithm by increasing computation. No added starts,
optimizer restarts or data per single prediction. An equal-temporal-prior
postfilter applies to existing baseline Pareto fronts OFFLINE; no second
search. This separates effects of prior and in-evolution structure selection.

Primary metric: selected_exact per system, second: truth_on_front. Report
paired effect by systems equal weight with cluster bootstrap + two-sided
exact signflip. With 9 physical laws min achievable p = 0.0039, but classes
may be correlated; do not claim universality or statistically independent
physics sampled randomly. Report wall seconds and timeouts, overfitting on
wave/heat, performance on TRUE u_xxx/u_xxxx explicitly.

**No further tuning of H10 complexity scale or H11 rule on these nine
validation outcomes**. If the result fails, retain it and design a genuinely
new hypothesis/data suite before further evaluation. Improvements to code
that are bug fixes must be documented and frozen into a new matched run
rather than silently combining different source SHAs.
