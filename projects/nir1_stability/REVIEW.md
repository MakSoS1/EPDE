# NIR-1 reviewer briefing — PROVISIONAL

**Scope:** coefficient instability, identifiability diagnostics and adaptive regularization  
**Branch:** [`nir1`](https://github.com/MakSoS1/EPDE/tree/nir1)  
**Recorded local commit:** `93b3edd39e5d373b76c7b975c0605aaac9633a9c`  
**S0 coverage:** 240/240 S0 method-trials recorded  
**Full EPDE search:** **NOT RUN**

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

## What these results cannot establish

- No S1/S2 heldout full-search conclusion is available: **NOT RUN**.
- Method counts share the same synthetic fixtures; they are not independent systems.
- `historical_proxy` does not reproduce the archive's 58/76 result.
- Ill-conditioned/weak-term failures must remain visible, not removed from the denominator.
- A comparison against current `chi2+vwsr` in real EPDE evolution is still required.

See [METHODS.md](METHODS.md), [RESULTS.md](RESULTS.md), and
`reports/s0/fixtures.jsonl.gz` (lossless gzip) for run-level verification.

## Independently recorded EPDE pilot and S0 job statuses

The full-search observations, failed-run denominators and failed GitHub Actions attempts are reported in [PILOT.md](PILOT.md). The fixed-candidate S0 table above must not be presented as full EPDE accuracy.
