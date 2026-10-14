# NIR-1 findings — PROVISIONAL

- **S0 candidate-level:** 240/240 S0 method-trials recorded; raw candidate rows: 930.
- **Invalid/unscorable candidate scores:** 80.
- **S1 full EPDE:** 120/120 completed and honestly reanalyzed; exploratory negative result.
- **S2 heldout and S3 transfer:** **NOT RUN**.
- **Historical vclog+swap (58/76 vs 45/76):** archival, **not reproduced**.
- **Numerical improvement over EPDE baseline:** **NOT ESTABLISHED**.

| Method | S0 trials | Exact support | Truth candidate ranked first | False deletions | False inclusions |
|---|---:|---:|---:|---:|---:|
| `baseline_ols` | 40 | 0 | 0 | 0 | 85 |
| `historical_proxy` | 40 | 33 | 35 | 0 | 15 |
| `hybrid` | 40 | 21 | 4 | 0 | 20 |
| `identifiability` | 40 | 33 | 4 | 0 | 8 |
| `instability` | 40 | 34 | 5 | 0 | 7 |
| `lasso` | 40 | 33 | 0 | 0 | 8 |

S0 is a **fixed-candidate regression/ranking test** on controlled, synthetically generated X and y. A success here is **not** successful EPDE evolutionary discovery. Methods did not consume truth labels; truth was used only to construct the diagnostic candidate set and score results.

**Status:** PROVISIONAL. Numerical rates must be qualified by full planned counts,
solver convergence and heldout split before making a scientific claim.

## Independently recorded EPDE pilot and S0 job statuses

The full-search observations, failed-run denominators and failed GitHub Actions attempts are reported in [PILOT.md](PILOT.md). The fixed-candidate S0 table above must not be presented as full EPDE accuracy.

## Completed exploratory S1 full search

GitHub Actions run [38076377893](https://github.com/MakSoS1/EPDE/actions/runs/38076377893) completed all 120 immutable identities: 6 systems × 4 methods × 5 paired optimizer seeds. All records were `ok`; there were no crashes, timeouts, unsupported cases, incomplete IDs or invalid checksums. Search code was frozen at `8d3aa947bc074000dee6d8acf96c4862eb74d81c`, while the unchanged records were reanalyzed with corrected analyzer snapshot `0246ed2fba663cb15b71ea68786aa17aa9a02f4f`.

| Method | Exact selected / 30 | Exact on front / 30 | Difference from baseline | Holm system-level p | Mean wall time |
|---|---:|---:|---:|---:|---:|
| `default` | 18 | 25 | — | — | 80.5 s |
| `nir1_criterion_only` | 4 | 19 | −46.7 pp | 0.375 | 79.8 s |
| `nir1_regulator_only` | 6 | 6 | −40.0 pp | 0.375 | 116.0 s |
| `nir1_combined` | 4 | 7 | −46.7 pp | 0.375 | 128.1 s |

All three proposed variants recovered fewer selected exact structures than the baseline. The exact system-level sign-flip p-values were 0.125, 0.1875 and 0.125; all three Holm-adjusted values were 0.375. Thus S1 provides no evidence of improvement. It is exploratory because six independent systems cannot attain Holm-adjusted p<0.05 for three comparisons even under perfect sign agreement. See the [verified S1 package](reports/actions_s1/README.md); do not use the archived pooled McNemar p-values.
