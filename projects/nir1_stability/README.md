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

## Quick verification

From the repository root, on a CPU machine with Python 3.12 and `uv`:

```bash
uv sync --project projects/pic --extra cpu --locked
uv pip install --python projects/pic/.venv/bin/python pytest
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/nir1-mpl projects/pic/.venv/bin/python -m pytest projects/nir1_stability/tests projects/pic/tests/test_runner_identity.py projects/pic/tests/test_metrics.py -q
```

Full-search regression with standard PIC budget (one fresh EPDE process):

```bash
OPENBLAS_NUM_THREADS=1 projects/pic/.venv/bin/python -m projects.nir1_stability.nir1.cli epde --dataset ode --variant default --seed 0 --data-seed 0 --output /tmp/nir1-ode-baseline.json
OPENBLAS_NUM_THREADS=1 projects/pic/.venv/bin/python -m projects.nir1_stability.nir1.cli epde --dataset ode --variant nir1_combined --seed 0 --data-seed 0 --output /tmp/nir1-ode-combined.json
```

The `--smoke` flag explicitly replaces the 16×5 PIC budget with 4×1 and
marks its record `nir1_smoke_only=true`; smoke accuracy is **never** S1
evidence. Full PIC experiments preserve baseline preprocessing, token pool,
evolution budget, seed and split. The standalone synthetic S0 elastic-net
regularizer and the real EPDE `nir1_adaptive` RFE per-term penalty are distinct
implementations, not falsely equated.

For immutable S0 Actions/local shards:

```bash
OPENBLAS_NUM_THREADS=1 projects/pic/.venv/bin/python -m projects.nir1_stability.nir1.cli plan --launch projects/nir1_stability/manifests/launch/s0-2026-10-10.yaml --output /tmp/nir1-planned.json
OPENBLAS_NUM_THREADS=1 projects/pic/.venv/bin/python -m projects.nir1_stability.nir1.cli shard --manifest /tmp/nir1-planned.json --id 0 --output /tmp/nir1-shard-0
```

Repeat `shard` for IDs `1`, `2`, `3`, then reconcile with `cli summarize`
using a directory that contains all per-shard JSON records. The output
enumerates every planned `ok`, `crash`, `timeout`, `unsupported` and
`incomplete` identity; infrastructure retries are bounded and preserved.

## Status and review

- [`REVIEW.md`](REVIEW.md) is the entry point for reviewers.
- [`RESULTS.md`](RESULTS.md) separates method diagnostics from actual full search.
- [`METHODS.md`](METHODS.md) documents assumptions, heldout gates and caveats.
- `reports/s0/fixtures.jsonl.gz` is complete raw fixed-candidate evidence.
- `reports/s0/ledger-records-local.tar.gz` retains the checksummed S0 ledger
  checkpoints, with `ledger-summary-local.json` for reconciliation.
- `reports/s1_pilot/` and `reports/real_smoke/` retain unaggregated real EPDE
  run records; never use one pair as a significant method comparison.

GitHub Actions workflows are restricted to this fork's `nir1` branch. The
inherited ITMO GitLab mirroring workflow has also been restricted on `nir1`:
research pushes do not trigger it. If GitHub reports a failed workflow run
with **zero created jobs**, no VM benchmarks ran; inspect repository Actions
settings/eligibility before claiming results from the branch workflow.
