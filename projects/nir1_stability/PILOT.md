# NIR-1 actual EPDE pilot — PROVISIONAL

**Scope:** genuine PIC/EPDE evolutionary searches; NOT the S0 fixed-candidate tests.
**Inferential status:** INSUFFICIENT independent systems/seeds for confidence intervals or superiority claims.
**SMOKE EXCLUDED:** 2 reduced-budget runs are not scientific trials.

## S0 frozen campaign reconciliation

- **236/240** successful S0 method trials; **4 crashes**, 0 timeouts, 0 unsupported, 0 incomplete.
- Every planned identity is counted, including invalid/singular solver cases.

## Measured full-search records

The success column uses the **PIC predefined compromise selector** (`metrics.success_selected`, truth-free min-max normalized objective sum). A separate Euclidean research selector is recorded in the raw JSON, but is not substituted after seeing truth. Source hashes here are local Git revisions used at execution; GitHub-hosted mirrored commits can have different commit IDs and must be matched by source contents.

| Dataset | Variant | Optimizer seed | Source revision | Status | PIC selected exact | Full fit (s) | Wall (s) |
|---|---|---:|---|---|---|---:|---:|
| `ode` | `default` | 0 | `3a1a5db` | ok | yes | 39.9 | 44.7 |
| `ode` | `nir1_combined` | 0 | `3a1a5db` | ok | no | 132.2 | 134.8 |
| `ode` | `nir1_criterion_only` | 0 | `3a1a5db` | ok | no | 52.5 | 57.4 |
| `ode` | `nir1_regulator_only` | 0 | `3a1a5db` | ok | no | 174.9 | 178.8 |
| `ode` | `default` | 1 | `3a1a5db` | ok | no | 39.8 | 43.3 |
| `vdp` | `default` | 0 | `3a1a5db` | ok | yes | 148.4 | 152.6 |
| `vdp` | `nir1_combined` | 0 | `3a1a5db` | ok | no | 299.2 | 306.0 |
| `vdp` | `nir1_criterion_only` | 0 | `3a1a5db` | ok | no | 120.4 | 123.8 |
| `vdp` | `nir1_regulator_only` | 0 | `3a1a5db` | ok | no | 232.1 | 236.6 |

## GitHub Actions evidence

- [Run 38073552748](https://github.com/MakSoS1/EPDE/actions/runs/38073552748): failure — **NO JOBS; no VM experiment executed**.
- [Run 38073552119](https://github.com/MakSoS1/EPDE/actions/runs/38073552119): failure — **NO JOBS; no VM experiment executed**.
- [Run 38075279279](https://github.com/MakSoS1/EPDE/actions/runs/38075279279): success — **1 job(s) observed; raw artifact verification required**.
- [Run 38075541420](https://github.com/MakSoS1/EPDE/actions/runs/38075541420): success — **6 jobs; raw records independently reconciled**.

## Interpretation and next gate

This small, mostly one-seed multi-system engineering pilot is not proof of a method effect. A slower or incorrect equation is a negative observation, not removed from statistics.
S2 heldout and S3 transfer cannot be advertised until the S1 finalist/budget are frozen and Actions (or another authorized isolated runner) executes all planned IDs.
The historical 58/76 count is not reproduced here.
