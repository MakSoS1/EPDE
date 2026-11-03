# CatBoost suggestions and public scientific examples

## Intent
Replace purely rule-based suggestions with a genuinely trained CatBoost portfolio
selector where independent equation-family evidence supports it. Keep manual
controls. Add immutable, clearly identified public examples to Results/Compare.

## Chosen approach
The user selected the previously proposed offline CatBoost surrogate approach.
Use the already frozen native campaign (eight synthetic inputs, seven families,
two noise levels, two seeds, four candidate policies per method). Reference laws
label historical results but are never model features. Hold out every seed, noise
level and configuration of an entire equation family together. Report every
failure and unavailable score, not just successful cases. No live training.

## Constraints
- Preserve existing EPDE/PDEComp scientific implementations and resource guards.
- Only select compatible, validated configurations from the frozen portfolio.
- Automatic activation requires positive family-held-out evidence and no loss of
  scoring coverage. Exploratory/negative evidence keeps rule-based settings.
- Domain and source/profile checks reject extrapolation and stale assets.
- Native scalar 1D PDE coverage is initial scope; ODEs, heavy inputs, unsupported
  protocols and out-of-domain numerical data retain explicitly explained fallback.
- CatBoost trains offline. Export numeric trees, test parity against native
  CatBoost, package checksummed assets; inference needs only NumPy.
- Public examples are explicitly curated builtin synthetic runs, not visitors'
  results. Preserve original equations, metrics, config, identity and provenance.
- Educational selection is not a representative leaderboard. Include limitations
  and unavailable metrics. Public views cannot delete or modify packaged records.
- The full experiment audit and grouped validation report are retained.

## Alternatives
Live hyperparameter trials are too expensive for automatic page rendering.
Rules alone do not meet the requested learned selection. The chosen offline
model plus honest fallback combines fast use with auditable quality evidence.
