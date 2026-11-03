# CatBoost and public results implementation plan

> Required execution: superpowers:executing-plans.

**Goal:** Ship trained, guarded suggestions and browsable public scientific runs.
**Architecture:** Frozen offline campaign -> family-held-out CatBoost validation ->
checksummed numeric model -> existing Suggest editors. A read-only public record
store feeds the existing Results/Compare renderers without private queue writes.
**Tech Stack:** Python, NumPy, CatBoost 1.2.8 (training), Streamlit, pytest, Chromium.
**Spec:** docs/superpowers/specs/2026-11-03-catboost-public-results.md

## Global constraints
No reference-law features. No training on the web server. No session data in
public examples. No optimistic score for missing/failed runs. No source hash
bypass. Keep native method settings and all manual overrides editable.

## Review focus
Near-border CatBoost numeric parity; incomplete campaign groups; changing model
assets without cache invalidation; invalid or out-of-domain arrays; public/private
navigation and deletion isolation.

## Tasks
- [ ] Audit saved campaign and finish the twelve missing EPDE requests. Retain
      original timeouts and all provenance; compare configuration/input keys,
      not regenerated floating-feature hashes across dependency environments.
- [ ] Test numeric predictor parity, held-out family isolation, strict activation,
      input enrichment and out-of-domain fallback. Fix only observed failures.
- [ ] Train model bundle and retain full audit plus per-input held-out choices;
      verify hashes and native/portable predictor parity before packaging.
- [ ] Test then integrate numerical descriptors and learned suggestions into the
      existing editors, explanations and saved advisory provenance.
- [ ] Test then add a read-only public store and selectors to Results/Compare;
      curate explicit synthetic examples with original metrics and provenance.
- [ ] Run local unit/AppTest suite and real Chromium views; publish to feature
      branch, run exact scientific/native CI and public browser checks; merge only
      verified feature code into demo branch, record any deployment limitation.

## Recovery verification
The complete 512-run campaign and trained models were recovered by archive checksum. A stale browser assertion accepted only rule-based provenance; it now accepts CatBoost only with an applied flag and a 64-character model checksum. Publication is gated by the complete scientific suite and actual four-method/Public results Chromium flows.
