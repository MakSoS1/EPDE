# Multisample Iteration 2 Implementation Plan

> **For agentic workers:** Use systematic debugging, test-driven changes and an independent final review. Steps use checkbox syntax.

**Goal:** Close reproduced multisample correctness gaps and deliver an executed, source-isolated before/after notebook with broader benchmark evidence.

**Architecture:** Preserve the independent `Trajectory`/`Domain` model. Define consistent snapshots and rectangular row geometry, harden shared fitting where evidence supports it, and run historical and current implementations in different processes. Keep notebook presentation separate from benchmark execution.

**Tech Stack:** Python, NumPy/SciPy, EPDE, pytest, Matplotlib, nbformat/nbclient/nbconvert.

**Spec:** `docs/superpowers/specs/2026-11-13-multisample-iteration2-design.md`.

## Global constraints

- Continue the dedicated research branch only; do not deploy or modify the Streamlit branch.
- Preserve numerical dependency versions while adding notebook execution dependencies.
- Keep all measured failures, rejected candidates and timeouts in the evidence.
- Do not claim that running an independent sparse selector is running EPDE.
- Use held-out whole experiments; no preprocessing or normalization from the held-out response.

## Review focus

- Raw versus cached versus weak observations after caller-side in-place edits.
- Shifted or perforated weight masks with the same row count as a valid rectangle.
- Finite weights/features whose naive sums or squared norms overflow or underflow.
- Source-import isolation and truthful version labels in notebook subprocesses.
- Saturated two-mode controls where additional experiments should confer no rank advantage.

## Tasks

- [x] Pin current upstream, first-iteration and demo revisions; inspect current branch cleanliness and notebook dependencies.
- [x] Independently reproduce omissions before editing, and distinguish inherited defects from first-iteration changes.
- [ ] Fix input snapshots and domain geometry in `epde/structure/domain.py`; add lifecycle and row-alignment regressions, update deliberately incorrect legacy expectations and run related domain/weak/pool tests.
- [ ] Harden pooled numerical helpers only after red evidence; test both the direct solve and relevant materialized-objective guards without promising unsupported extreme-scale stability.
- [ ] Audit demo tests, transfer generators with exact provenance, and validate new actual-EPDE cases against their original analytic fixtures.
- [ ] Freeze production source, run the bounded multi-case historical/current campaign, preserve all outcomes and a same-source mean ablation.
- [ ] Create and execute the Russian notebook and HTML export with genuine historical/current subprocesses, a third iteration-1 column for newly corrected defects, and explicit archival-result labels.
- [ ] Run the complete unit suite, relevant PIC/demo suites and the repository pytest command; compare with known baseline failures.
- [ ] Independently review the final patch and executed notebook, verify artifact links/fingerprints, and save the branch without creating a PR.

## Evidence already established

- Demo-selected scientific and validation-contract tests: 79 passed and 30 subtests passed.
- Initial adversarial audit: six failed expectations and one valid-mask control passed; field/geometry mismatches also reproduce on upstream.
- Notebook dependency installation constrained all 149 pre-existing distributions and changed none of their versions.
- Demo H8 supplies ten four-initial-condition generators; its original selector is controlled fixed-target sparse regression, not unrestricted EPDE. Its two-mode audit provides a useful no-rank-benefit control.
