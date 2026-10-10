# NIR-1: coefficient instability and adaptive regularization in EPDE

This is a separate, opt-in scientific research workspace based on
`MakSoS1/EPDE` branch `pic/review-ready` at
`bd6babc8d2cf9b69bcb878ead7048a962ab08cf8`.
Its working branch is **`nir1`**. Nothing here changes the ITMO upstream or
the existing PIC benchmark defaults.

Read the preregistered protocol at
[`../../docs/research/NIR1_EPDE_Research_Protocol_2026-10-10.md`](../../docs/research/NIR1_EPDE_Research_Protocol_2026-10-10.md)
and the technical design at
[`../../docs/superpowers/specs/2026-10-10-epde-nir1-implementation-design.md`](../../docs/superpowers/specs/2026-10-10-epde-nir1-implementation-design.md).

Run identity includes the source revision, dataset and configuration hashes,
data/optimizer seeds and fixed split. A missing run is **never** a success.
Exact weighted-Gram reuse is the default optimization; subsampling and
reduced evolution budgets are **not** silently enabled.

Results in this directory must distinguish controlled fixed-candidate
experiments from full EPDE discovery runs. Historic comparison counts are
external references until replayed against this branch.
