# Multisample implementation plan

> **For agentic workers:** use test-driven development, independent file ownership, and a final fresh review. Steps use checkbox syntax for tracking.

**Goal:** Make independent-sample input reliable and add a verifiable joint-coefficient search mode.

**Architecture:** Preserve trajectories and grids throughout preprocessing. Share one weighted regression implementation between plain and assignment search; retain historical mean behavior by default.

**Tech stack:** Python, NumPy, SciPy, existing EPDE operators, pytest, CPU PyTorch.

**Spec:** `docs/superpowers/specs/2026-11-13-multisample-design.md`.

## Global constraints

- Base SHA is `8690035e90b234b34ed5a2fb0bb8e4a814dcbdf7`.
- Isolated fork branch only; no upstream changes or PR.
- Weak-form mathematical implementation remains unchanged.
- No silent invalid-sample filtering, invented results, hidden downsampling, or unequal A/B budgets.

## Review focus

- Joint tables must predict exactly the objectives of the materialized common equation.
- Rank-deficient individual samples may jointly identify coefficients; instability must remain honest.
- Changing trajectory order must not change mathematical acceptance.
- Resampling a trajectory must not change its total fitting weight under `samples`.
- Rebuilds and repeated fits must not reuse another sample's grids, smoothing, or evaluator values.

## Tasks

- [x] Baseline: run original `pytest tests/unit`, preserve output and environment.
- [x] Domain/cache: reproduce shared-PDE-domain, nonzero time axis, repeated smoothing and evaluator cache failures; fix `domain.py`/`evaluators.py`; run new and related regressions.
- [x] Interface/config: add joint settings and fail-loud validation; replace the stale multisample class with a current-API adapter; first add and run failing API tests.
- [x] Joint regression: add `operators/common/multisample.py`; wire plain/assignment common fits, rank and screen weighting; first run failing numerical and operator tests.
- [x] Proposals: let joint fill-and-prune pool complete row blocks with the same fit weights; test against an independently computed Gram.
- [x] Benchmarks: fixed-support and evolutionary runs, independent held-out cases, clean/noisy and strong/weak evidence, exact seeds and budgets, machine-readable results.
- [x] Review: inspect all diffs, run unit and repository test commands, investigate failures, fresh independent review, document limitations.
- [x] Deliver: commit and save the dedicated fork branch, provide reproducible report and commands.

## Decisions during execution

- The request authorizes completing reversible implementation and tests; no additional design-approval pause is needed.
- Retain the mean mode by default to preserve historical runs. Joint VWSR/repair is refused rather than approximated by a union of locally pruned supports.
- Preserve each existing discrepancy definition; fit weighting does not silently redefine WAPE or per-grid instability.

## Final verification evidence

- Baseline unit suite: 2603 passed, five historical solver digest mismatches.
- Final unit suite: 2703 passed, the same five failures; all baseline case statuses unchanged.
- Added 96 targeted multisample cases plus four automatic module collection checks; all passed.
- PIC project: 88 tests and 23 subtests passed on both checkouts.
- Benchmark: 98 final jobs, 97 with models and one expected refused fixed support; no execution errors or timeouts.
- Whole-repository pytest still has the three reproduced baseline collection blockers; no blanket green-suite claim.
- Review findings (nonphysical sparsity refit and destructive/ID-dependent constancy) corrected and retested.
- Complete evidence and limits: `projects/multisample/VERIFICATION.md`.
