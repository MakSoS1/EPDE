# Frozen real-EPDE S1 cohort — completed and reanalyzed

- **Research Actions run:** [38076377893](https://github.com/MakSoS1/EPDE/actions/runs/38076377893), completed successfully on 10 October 2026.
- **Search code executed by all VMs:** `8d3aa947bc074000dee6d8acf96c4862eb74d81c`.
- **Corrected analysis snapshot:** `0246ed2fba663cb15b71ea68786aa17aa9a02f4f`.
- **Statistical fix commits:** system-level exact sign-flip `e71ee64c3440e724b2b55c121054bcd693917033`; Holm over `cluster_signflip_p` `6602e35c00aea06fe418edd600b70e2864204bbd`.
- **Immutable expanded manifest:** [`planned.json`](planned.json), `manifest_sha=7993e0811e9cc9f14f7a3ff79aeab76d618b4144d8a002c9ae1b45cf413ce4e7`.
- **Design:** 6 systems × 4 factorial methods × 5 paired optimizer seeds = 120 full EPDE searches with the standard PIC 16×5 budget.

## Integrity verification

All six downloaded Actions ZIP archives matched the SHA256 digest returned by GitHub. The four shard archives contained exactly the 120 unique IDs in the immutable manifest: no missing or extra IDs, 120 internally checksummed records, and the same `manifest_sha` in every record. Final statuses were 120 `ok`, 0 `crash`, 0 `timeout`, 0 `unsupported`, 0 `incomplete`; no bounded retry was needed.

Artifact IDs and verified archive digests are recorded in [`ARTIFACT_SHA256SUMS.txt`](ARTIFACT_SHA256SUMS.txt). The workflow's original aggregate is retained unchanged under [`archive/run-38076377893-original/`](archive/run-38076377893-original/) for provenance only. Its pooled run-level McNemar p-values and Holm correction are invalid for inference because five optimizer seeds from one system are not independent systems.

## Corrected S1 result

The unchanged raw records were reanalyzed with the corrected code. The machine-readable result and review table are in [`corrected/S1_SUMMARY.json`](corrected/S1_SUMMARY.json) and [`corrected/S1_SUMMARY.md`](corrected/S1_SUMMARY.md). A second independent invocation produced byte-identical files; the corrected JSON SHA256 is `0a3570f81d718f843412c4adf1f054c30b091cf70eb7f6330b61cd252d558d34`.

| Method | Exact selected / 30 | Difference from baseline | 95% system-cluster bootstrap CI | Exact system sign-flip p | Holm p | Mean wall time |
|---|---:|---:|---:|---:|---:|---:|
| `default` | 18 (60.0%) | — | — | — | — | 80.5 s |
| `nir1_criterion_only` | 4 (13.3%) | −46.7 pp | [−80.0, −13.3] pp | 0.1250 | 0.375 | 79.8 s |
| `nir1_regulator_only` | 6 (20.0%) | −40.0 pp | [−80.0, +0.1] pp | 0.1875 | 0.375 | 116.0 s |
| `nir1_combined` | 4 (13.3%) | −46.7 pp | [−83.3, −10.0] pp | 0.1250 | 0.375 | 128.1 s |

The baseline also placed the true structure on the Pareto front in 25/30 searches, versus 19/30, 6/30 and 7/30 for criterion-only, regulator-only and combined variants. S1 therefore provides no evidence that the proposed variants improve the frozen baseline; descriptively, all three recovered fewer exact selected structures and two were slower.

The per-system 5-seed breakdown and scientific interpretation are recorded in [`../../S1_VERIFIED_RU.md`](../../S1_VERIFIED_RU.md).

This is an exploratory six-system result, not a confirmatory significance claim. With six independent systems, the smallest possible two-sided exact sign-flip p-value is 0.03125; with three prespecified comparisons the smallest possible Holm-adjusted p-value is 0.09375. S2/S3 remain unrun and transfer to noisy or real systems is not established.

