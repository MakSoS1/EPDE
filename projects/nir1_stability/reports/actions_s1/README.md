# Frozen real-EPDE S1 cohort — execution in progress

- **Research Actions run:** [38076377893](https://github.com/MakSoS1/EPDE/actions/runs/38076377893).
- **Exact source commit executed by the VMs:** `8d3aa947bc074000dee6d8acf96c4862eb74d81c`.
- **Immutable expanded S1 manifest:** [`planned.json`](planned.json); manifest hash
  `7993e0811e9cc9f14f7a3ff79aeab76d618b4144d8a002c9ae1b45cf413ce4e7`.
- **Design:** 6 systems × 4 factorial methods × 5 paired optimizer seeds = **120 planned full EPDE searches**. Standard PIC population/epoch budget; no short smoke runs counted.
- **Resources:** 4 GitHub-hosted Ubuntu CPU shards, 300-minute soft deadline and 340-minute hard limit; each identity up to 30 minutes. Missing results stay `incomplete` and incomplete jobs can be rerun from checksum-verified artifacts.
- **Official frozen-manifest GitHub artifact:** ID `11679060411` with SHA256 `5d84a2726a3f5dabe5be36ca8b71846b5f095b493803154ce60b4c467d3aefff`.

**As of launch:** `prepare` succeeded and four shard VM jobs began actual evolution. **No scientific S1 aggregate exists yet.** Once shards finish, the `aggregate` job writes `summary.json`, `S1_SUMMARY.json` and `S1_SUMMARY.md`, keeping all failures in the denominator and blocking paired inference on missing/unsupported pairs.

The old 9-run local two-system feasibility pilot is separately recorded in
[`../../PILOT.md`](../../PILOT.md). Do not mix its source hashes or its data
into the frozen 120-run S1 cohort.
