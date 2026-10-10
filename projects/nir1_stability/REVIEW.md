# NIR-1 reviewer briefing — PROVISIONAL

**Scope:** coefficient instability, identifiability diagnostics and adaptive regularization  
**Branch:** [`nir1`](https://github.com/MakSoS1/EPDE/tree/nir1)  
**Recorded local commit:** `93b3edd39e5d373b76c7b975c0605aaac9633a9c`  
**S0 coverage:** 240/240 S0 method-trials recorded  
**Full EPDE search:** **S1 COMPLETE, EXPLORATORY NEGATIVE RESULT**

## Verified from available evidence

S0 is a **fixed-candidate regression/ranking test** on controlled, synthetically generated X and y. A success here is **not** successful EPDE evolutionary discovery. Methods did not consume truth labels; truth was used only to construct the diagnostic candidate set and score results.

| Method | S0 trials | Exact support | Truth candidate ranked first | False deletions | False inclusions |
|---|---:|---:|---:|---:|---:|
| `baseline_ols` | 40 | 0 | 0 | 0 | 85 |
| `historical_proxy` | 40 | 33 | 35 | 0 | 15 |
| `hybrid` | 40 | 21 | 4 | 0 | 20 |
| `identifiability` | 40 | 33 | 4 | 0 | 8 |
| `instability` | 40 | 34 | 5 | 0 | 7 |
| `lasso` | 40 | 33 | 0 | 0 | 8 |

## Completed exploratory S1

The immutable 120-run full-search cohort completed with 120 `ok` records and no missing, corrupt, crashed, timed-out or unsupported identity. The frozen baseline selected an exact structure in 18/30 searches; criterion-only, regulator-only and combined variants achieved 4/30, 6/30 and 4/30. Correct independent-system exact sign-flip p-values were 0.125, 0.1875 and 0.125; all Holm-adjusted p-values were 0.375. The archived workflow aggregate's pooled McNemar p-values are diagnostic only and must not be cited.

See the [verified S1 evidence and corrected report](reports/actions_s1/README.md).

## What these results cannot establish

- S1 does not establish an improvement; S2 heldout and S3 transfer studies were not run.
- Method counts share the same synthetic fixtures; they are not independent systems.
- `historical_proxy` does not reproduce the archive's 58/76 result.
- Ill-conditioned/weak-term failures must remain visible, not removed from the denominator.
- Six independent systems are insufficient for Holm-adjusted p<0.05 across three comparisons; confirmatory inference needs a larger preregistered independent-system cohort.

See [METHODS.md](METHODS.md), [RESULTS.md](RESULTS.md), and
`reports/s0/fixtures.jsonl.gz` (lossless gzip) for run-level verification.

## Independently recorded EPDE pilot and S0 job statuses

The full-search observations, failed-run denominators and failed GitHub Actions attempts are reported in [PILOT.md](PILOT.md). The fixed-candidate S0 table above must not be presented as full EPDE accuracy.
