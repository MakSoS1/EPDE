# Live full causal ablation — launched 2026-10-14

Research branch: `research/nir1-mdl-factorial-2026-10-14`
Full preregistration: [MDL_CAUSAL_ABLATION_2026-10-14.md](MDL_CAUSAL_ABLATION_2026-10-14.md)
Runner: [nir1/mdl_factorial_runner.py](nir1/mdl_factorial_runner.py)

## Actual GitHub Actions IDs (created and preflight confirmed)

| optimizer seed | run | count | preflight | workflow snapshot |
| --- | --- | --- | --- | --- |
| 0 | [38472631487](https://github.com/MakSoS1/EPDE/actions/runs/38472631487) | 252 full + preflight | passed | `a2d0f22db1c87b8dd197bf024cc80aaa8ed6051c` |
| 1 | [38472636721](https://github.com/MakSoS1/EPDE/actions/runs/38472636721) | 252 full + preflight | passed | `44e7d8d65bd67ae6ac5701a63b3a9e9dbc28c604` |
| 2 | [38472639768](https://github.com/MakSoS1/EPDE/actions/runs/38472639768) | 252 full + preflight | passed | `9db33deb450a313708652404f1320c8993459bb8` |

All three preflight smoke checks passed, and full search matrix jobs started. Their Git SHAs differ only by added workflow files, not by model code. GitHub Actions results are *not complete as of this launch log*; do not cite success rates before reconciling 756 identities.

## Observed diagnostic during initial full jobs

The early `ac/mdl_utopia/seed=0` failure is a **real EPDE search failure** with the original unpatched H9/H10 guard:

```text
RuntimeError("NIR1 H9: no time derivative survived the equation search")
```

This is an anticipated no-repair arm failure, **not** a dependency/runner problem. Keep as `failure`, do not retry selectively. The run's `native-full.json` and traceback were uploaded to artifact 11853260722.

## Scientific rule for reading outcomes

- Primary causal contrast: `mdl_min_repair - no_mdl_min_repair` on **paired 36 systems ×3 seeds**.
- Distinguish `success_selected` and `success_front`; count crashes and timeouts as failed recovery.
- Compare outcomes at the **physical-system cluster** level. Seed replicates are not independent new laws.
- All 36 laws were previously seen during earlier EPDE/H9-H14 development; these are not fresh heldout.
- The original ITMO upstream is NOT one of the seven arms.
- E1 acceleration disabled. No changes to upstream ITMO, `main`, or `nir1`.

Every workflow step checks out the event commit by default, so later branch documentation edits do not change the already-started experimental snapshots.
