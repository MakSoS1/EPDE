# NIR-1 S1 — auditable full-search comparison

Frozen manifest: `7993e0811e9cc9f14f7a3ff79aeab76d618b4144d8a002c9ae1b45cf413ce4e7`; source SHA: `8d3aa947bc074000dee6d8acf96c4862eb74d81c`.
Planned full EPDE runs: **120**. Inference gate: **PROVISIONAL_S1_ONLY**.

Every planned ID is counted; a timeout, algorithm failure, or missing record is never classified as an exact recovery.

## Methods, all planned identities

| Method | Planned | OK | Crash | Timeout | Unsupported | Incomplete | PIC selected exact / planned | Mean wall(s), OK only |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `default` | 30 | 30 | 0 | 0 | 0 | 0 | 18/30 | 80.5 |
| `nir1_criterion_only` | 30 | 30 | 0 | 0 | 0 | 0 | 4/30 | 79.8 |
| `nir1_regulator_only` | 30 | 30 | 0 | 0 | 0 | 0 | 6/30 | 116.0 |
| `nir1_combined` | 30 | 30 | 0 | 0 | 0 | 0 | 4/30 | 128.1 |

## Paired vs PIC baseline (method minus baseline)

| Method | Scored / expected | Unresolved | Unsupported pairs | Difference (pp) | 95% system-cluster CI (pp) | Holm system-level p | Gate |
|---|---:|---:|---:|---:|---|---:|---|
| `nir1_criterion_only` | 30/30 | 0 | 0 | -46.7 | [-80.0, -13.3] | 0.375 | RECORDED_DESCRIPTIVE_WITH_CI |
| `nir1_regulator_only` | 30/30 | 0 | 0 | -40.0 | [-80.0, +0.1] | 0.375 | RECORDED_DESCRIPTIVE_WITH_CI |
| `nir1_combined` | 30/30 | 0 | 0 | -46.7 | [-83.3, -10.0] | 0.375 | RECORDED_DESCRIPTIVE_WITH_CI |

## Integrity and limits

- Status totals: `{'ok': 120, 'timeout': 0, 'crash': 0, 'unsupported': 0, 'incomplete': 0}`.
- Invalid or corrupted records: **0** (not silently dropped from planned denominators).
- The recovery measure is the frozen PIC truth-free compromise selector, scored offline against known truth.
- A system-clustered interval and exact system-signflip test only appear when all planned pairs are verifiable and at least 3 systems are present.
- `PROVISIONAL_S1_ONLY` does not imply statistical significance, transfer to real noisy data or completion of S2/S3 heldout validation.
