# NIR-1 findings — PROVISIONAL

- **S0 candidate-level:** 240/240 S0 method-trials recorded; raw candidate rows: 930.
- **Invalid/unscorable candidate scores:** 80.
- **S1 full EPDE, S2 heldout, S3 transfer:** **NOT RUN**.
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
