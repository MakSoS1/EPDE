# NIR-1 actual EPDE pilot — PROVISIONAL

**Scope:** genuine PIC/EPDE evolutionary searches; NOT the S0 fixed-candidate tests.
**Inferential status:** INSUFFICIENT independent systems/seeds for confidence intervals or superiority claims.
**SMOKE EXCLUDED:** 2 reduced-budget runs are not scientific trials.

## S0 frozen campaign reconciliation

- **236/240** successful S0 method trials; **4 crashes**, 0 timeouts, 0 unsupported, 0 incomplete.
- Every planned identity is counted, including invalid/singular solver cases.

## Measured full-search records

| Dataset | Variant | Optimizer seed | Status | Selected exact | Full fit (s) | Wall (s) |
|---|---|---:|---|---|---:|---:|
| `ode` | `default` | 0 | ok | yes | 57.8 | 62.0 |
| `ode` | `nir1_combined` | 0 | ok | no | 170.3 | 180.6 |

## GitHub Actions evidence

- [Run 38073552748](https://github.com/MakSoS1/EPDE/actions/runs/38073552748): failure — **NO JOBS; no VM experiment executed**.
- [Run 38073552119](https://github.com/MakSoS1/EPDE/actions/runs/38073552119): failure — **NO JOBS; no VM experiment executed**.

## Interpretation and next gate

A single paired system/seed is an engineering smoke/pilot, not proof of a method effect. A slower or incorrect equation is a negative observation, not removed from statistics.
S2 heldout and S3 transfer cannot be advertised until the S1 finalist/budget are frozen and Actions (or another authorized isolated runner) executes all planned IDs.
The historical 58/76 count is not reproduced here.
