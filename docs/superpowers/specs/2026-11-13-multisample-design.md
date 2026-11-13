# Multisample on domain_refactor

## Scope and baseline

Base: `ITMO-NSS-team/EPDE`, `domain_refactor`, `8690035e90b234b34ed5a2fb0bb8e4a814dcbdf7`.
Improve the library's common-equation search from independent trajectories. The weak-form engine is retained; compatibility is tested. Work is isolated in the user's fork, with no upstream or existing-branch update and no pull request.

## Approaches and decision

1. Repair only the legacy `EpdeMultisample` wrapper. This restores an entry point but leaves coefficient averaging and first-sample guards untouched.
2. Replace all regression/sparsity paths at once. This changes historical experiments and requires a new joint sparse regularizer and stability interpretation.
3. Add an explicit joint fit for the current fixed-support `plain` and subset `assignment` paths, retain `mean` as the compatibility default, and repair sample handling for both. This is the selected implementation decision: it isolates a measurable mathematical change without claiming unsupported VWSR behavior.

## Contracts

- Independent preprocessing and derivative stencils; flatten/stack only evaluated term rows.
- One physical coefficient vector per candidate, fitted with weighted least squares across all samples in joint mode.
- `objectives.multisample_mode`: `mean` (default) or `joint`; joint requires `plain` or `assignment` and a solver-free multiobjective search.
- `objectives.multisample_weighting`: `samples` (default) gives each sample's existing positive row weights unit total mass; `points` keeps existing row weights. It governs the fit and proposal screens. Existing discrepancy definitions remain explicit and unchanged: WLS is not claimed to minimize WAPE.
- Joint rank is tested on the combined weighted design. A locally singular trajectory is not a reason to reject a globally identifiable support.
- Candidate tables and materialized equations evaluate the same common coefficient vector and the same discrepancy aggregation.
- Instability remains the existing per-grid diagnostic on the common active support. It is not proof of between-sample coefficient agreement or statistical identification; held-out residuals and rank diagnostics are reported separately.
- No silent omission of nonfinite samples, incompatible variables, duplicate sample IDs, or conflicting domain IDs.
- The legacy class becomes a thin adapter to the current search, accepting `(grids, fields)` samples; no duplicate optimization implementation.

## Verification

Record the unmodified unit-suite baseline. Reproduce each fixed bug before the fix. Test independent/rebuilt preprocessing, shared and unequal 2-D grids, time-axis placement, input validation, multiple variables, cached evaluators, sample permutation, weighting/density, constant support, contradictory laws, strong and weak rows, and both search modes.

Benchmark fixed-support recovery separately from stochastic discovery. Use fixed seeds and independent held-out trajectories. Include analytically differentiable ODEs, a coupled ODE, and a reaction-diffusion PDE whose two individually rank-one records are jointly identifiable. Record failures and timeouts, exact budgets, versions, and raw measurements; do not describe analytical coefficient recovery as full symbolic search success.
