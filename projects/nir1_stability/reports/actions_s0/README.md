# Frozen S0 replay — GitHub Actions VM evidence

- Fork/branch: `MakSoS1/EPDE`, `nir1` only; upstream ITMO unchanged.
- [Workflow run #38075541420](https://github.com/MakSoS1/EPDE/actions/runs/38075541420) on remote commit
  `ada3f84c801d6e257d08f042f0c7300b01e81f63`.
- Actions finished **successfully**: `prepare`, shards 0–3, and `aggregate`.
- Frozen `planned.json`: manifest SHA256
  `f50e663f9f0717e8a4354119643c2131196caac74fde974b0cfe8f3b06c9c249`.
- `summary.json`: **240 planned**, **236 ok**, **4 algorithm crashes**, **0 missing**,
  **0 invalid/checksum failures**, **0 timeouts**. The 4 crashes were not retried or discarded.
- `shard-{0,1,2,3}.zip` are the **original Actions artifact archives**,
  each containing 60 checksummed per-identity records. The local independent
  recomputation using `cli summarize` matched `summary.json` **byte-for-byte**.

| GitHub artifact ID | Local archive | SHA256 (matches GitHub artifact digest) |
|---|---|---|
| 11677474597 | `shard-0.zip` | `6b3d2f3af32fa155bd98e9396f0d8d5db4f4b44f6029f6e327da52a3d196814d` |
| 11678239243 | `shard-1.zip` | `6dae741653d2158e78763937376bbc45c542b3757028f2c8a79af3422c0660cb` |
| 11677894815 | `shard-2.zip` | `6ec46fbedf6640532fe1ef7910f9500433088ec2196edee8b5233927d3de028e` |
| 11678154325 | `shard-3.zip` | `05594aee6dc9c24640ddfa6e30c632327da52a3d196814d` |

To re-audit the archives without network access, unpack all four into one
temporary directory (retaining the `shard-X/` directories) and run from the
repository root:

```bash
python -m projects.nir1_stability.nir1.cli summarize \
  --manifest projects/nir1_stability/reports/actions_s0/planned.json \
  --records /path/to/unpacked-shards/ \
  --output /tmp/nir1-actions-s0-recomputed.json
```

The same S0 design was previously run locally on another source revision, also
236/240; they are two separately identified campaigns, **not 480 independent
scientific samples**, because their fixed candidates and seeds are shared.
