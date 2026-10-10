# NIR-1 methods and limitations

1. T1–T8 controlled synthetic arrays with independent signal, target-noise and derivative-error RNG streams.
2. Weighted sufficient statistics: G=XᵀWX, b=XᵀWy, and exact support submatrix reuse.
3. Coefficient diagnostics: per-environment fitted-coefficient variance minus estimated iid uncertainty, scaled by normalized coefficient energy; SVD-based per-term identifiability Q.
4. Candidate-level OLS, LASSO, instability-weighted and identifiability-weighted elastic net. Tuning here uses fixed preregistered λ; it has not been optimized on heldout truth.
5. Fixed truth/decoy candidate sets are constructed by the simulation oracle, but data-only metrics and sparse fits read no truth. No EPDE evolution occurs in S0.
6. Historical vclog proxy is not the main-branch multi-axis correction; archive's 58/76 is an external reference only.
7. S1 is a completed exploratory six-system evaluation with five paired optimizer seeds per method. Confirming improvement would require a separately frozen S2 with more independent systems, uncertainty estimates and all recorded failures; repeated seeds do not increase the independent-system sample size.
8. E1 acceleration uses content-addressed Gram cache, parity checks and SVD/reference fallback. E2 approximate speedups are disabled by default.
9. **Paired statistics updated 10 Oct 2026:** repeated optimizer seeds are NOT independent physical systems. Equal-weight per-system paired success differences are evaluated with whole-system clustered bootstrap and an exact two-sided system-level sign-flip test (for at most 18 systems), then Holm across the three prespecified S1 contrasts. The pooled run-level McNemar probability is diagnostic only, not eligible for scientific inference. At six independent systems the smallest unadjusted two-sided p is 0.03125 and three-comparison Holm p cannot reach 0.05. The frozen S1 raw records were reanalyzed with corrected code snapshot `0246ed2fba663cb15b71ea68786aa17aa9a02f4f`; the original aggregate is archival only.
10. Under errors in predictor derivatives, reported iid sampling variance is an **assumption check**, not calibrated confidence. `NOT RUN` is never zero accuracy.
