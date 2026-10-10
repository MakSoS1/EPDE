# NIR-1 v2 — front selection after S1 failure analysis

**Research status:** exploratory development hypothesis; independent prospective verification pending.
**Source:** MakSoS1/EPDE `nir1`. **Never change ITMO upstream.**

## Observed failure decomposition (all 120 immutable S1 full-search records)

| Method | Truth appeared on Pareto front | PIC selected truth |
|---|---:|---:|
| `default` | 25 / 30 | 18 / 30 |
| `nir1_criterion_only` | 19 / 30 | 4 / 30 |
| `nir1_regulator_only` | 6 / 30 | 6 / 30 |
| `nir1_combined` | 7 / 30 | 4 / 30 |

The first structural bottleneck is **Pareto-front selection**, whereas the second is **over-pruning prior to selection**. These mechanisms must be addressed separately.

### Post-selection hypothesis H7: sparsefront

For each candidate system with `m` equations, the first, third, etc. entries in its objective vector are dimensionless PIC discrepancy, and the remaining entries are instability. After a fixed EPDE search, rank candidates without viewing truth:

[
J(E) = \frac{1}{m}\sum_{k=1}^m L_k(E) + 0.003 N_{\mathrm{rhs}}(E) +
0.001\sum_{d \in D_{\mathrm{rhs}}(E)}\max(0,\operatorname{order}(d)-2).
]

Here `N_rhs` counts nonzero equation right-hand-side terms (EPDE prints these to the left of `=`), and `D_rhs` contains derivative factors in those terms. This is an **explicit, small complexity prior**. The second PIC axis remains active during evolutionary discovery; it is not used in this v2 final choice. It is not a new coefficient-instability metric or a new regularizer.

The constants above were examined among other possibilities on **the existing six S1 systems** after S1 outcomes were known. Their selection is **post-hoc / development-tuned**. The retrospective result of the *same* baseline fronts was **23/30 versus 18/30** for the old compromise selector, a +16.7 percentage-point difference (five cases). This is not independent evidence, and no p-value for this contrast is presented as confirmatory.

### Required validation

- Preserve the full Pareto front and objectives for every run; record `metrics_pic_original` with the unmodified PIC verdict and `metrics` with the new strategy's verdict.
- For direct paired evaluation, the **same** EPDE evolution and front can be scored by both selectors. It is unnecessary to double the evolution budget merely to test a deterministic post-selector, although having `default` runs as an independent fidelity control is useful.
- Evaluate an untouched set of systems with exactly the same 16 × 5 EPDE budget and five optimizer seeds. Do not tune the weights on these validation results.
- Analyze both exact structure (PIC truth only for *scoring*) and wall clock time. Claims about out-of-distribution superiority require multiple systems and S2/S3, not one seed.
- Avoid disclosing heldout truth to any branch used to select candidates.
- Preserve all failures and incomplete runs and the immutable commit/config/manifest identities.

### Limitations and follow-up

S1 also establishes that simply giving **low-identifiability columns a larger L1 penalty** can destroy the true support. Future RFE work should protect low-identifiability terms unless a separate, data-only residual increase test justifies deletion. The separate regularization hypothesis requires independent development and heldout tests; it is not claimed to have been solved by `nir1_sparsefront`.

**Success gate:** improvements must be measured on systems and data seeds that did not participate in selecting the weights. If no improvement occurs, publish the negative result without changing the rules after seeing the heldout outcome.
