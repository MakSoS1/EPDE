# NIR-1 v2 — identifiability-protected regularization hypothesis

**Pre-experiment proposal, 10 October 2026.** This is development of the experimental EPDE variant, not a confirmed improvement.

## S1 reason for trying this

Full EPDE 120-run S1 revealed candidate-generation loss:
- baseline: truth on front 25/30,
- initial instability-weighted regularizer: truth on front 6/30,
- initial regularizer+objective: 7/30.

The original NIR1 score \(\tfrac12[E+(1-Q)]\) treats **a non-identifiable but physically correct term** as maximally suspicious, even when its observed heterogeneity is consistent with sampling uncertainty. Consequently it can be penalized into disappearance.

## Mechanism

Define \(E_j\) as the existing noise-excess heterogeneity statistic, and \(Q_j\in[0,1]\) as the normalized unique Gram energy. New candidate penalty:

\[
\rho_j^{\mathrm{protected}} = E_j Q_j.
\]

When \(Q_j\) is low, the column is not individually identifiable; a high instability penalty is not justified by *that* statistic alone. When \(Q_j\) is close to one, its excess heterogeneity is allowed to affect selection.

This is deliberately **not a guarantee** of correctness: protecting low-identifiability terms could retain decoys. The score does not estimate causal truth, and the iid sampling variance assumption for derivative errors remains uncalibrated.

Implementation:
- `epde.operators.common.survival.nir1_protected_scores`, a shared estimator;
- `epde.interface.search_config.ObjectivesConfig` opt-in validation;
- `Nir1AdaptiveSparsity` opt-in RFE;
- `epde_adapter.resolve_research_variant("nir1_protected_regulator")`;
- default EPDE and previous variants remain unchanged.

## Development pilot to launch

Five already observed S1 systems (`ode, vdp, duffing, ac, burgers`) × 2 methods (`default, nir1_protected_regulator`) × 3 paired seeds = 30 full-budget EPDE runs. Inference is exploratory because the systems were already included in S1. Analyze on the *planned denominator*, including timeouts and crashes; compare truth-on-front, truth-selected, time and total RFE work. This pilot must not be conflated with validation S1b for the separately developed sparsefront post-selector.

If the method loses structure, keep the negative result and reconsider an explicit data-only residual-increase deletion gate. If it helps, freeze the algorithm and test against untouched systems before making any superiority claim.
