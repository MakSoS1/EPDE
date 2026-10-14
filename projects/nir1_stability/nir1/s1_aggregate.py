"""Prerecorded, failure-aware S1 analysis of immutable full EPDE searches.

The truth-free PIC compromise is selected during each EPDE run. Ground truth
is read only here, from the immutable per-run ``metrics`` object. This code
never tunes search hyperparameters or excludes inconvenient solver failures.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from .campaign import resume_plan
from .metrics import holm_adjust, paired_effect_ci
from .records import STATUSES, read_record


def analyze_s1(manifest: Mapping[str, object], paths: Sequence[Path]) -> dict[str, object]:
    """Analyze all S1 run IDs (missing and crashed included in denominators).

    A pre-registered comparison only gets a paired system-clustered interval
    when *all* expected pairs are verifiable, no pair is unsupported, and
    three or more independent systems exist. Algorithm crashes count as
    failures, while an unfinished/infrastructure pair blocks inference.
    """
    if manifest.get("stage") != "S1":
        raise ValueError("Only S1 full-search manifests are compatible with paired S1 analysis")
    paths = [Path(p) for p in paths]
    ledger = resume_plan(manifest, paths)
    latest = {}
    for path in paths:
        try:
            record = read_record(path)
        except (ValueError, OSError, json.JSONDecodeError):
            continue  # Already explicit in ledger.invalid_records; never success.
        run_id = str(record["run_id"])
        if (run_id not in latest or
                int(record.get("attempt", 0)) > int(latest[run_id].get("attempt", 0))):
            latest[run_id] = record

    planned = list(manifest["runs"])
    methods = list(dict.fromkeys(str(run["method"]) for run in planned))
    if "default" not in methods:
        raise ValueError("S1 requires a frozen default baseline for paired comparisons")
    per_method = {name: {"planned": 0, "selected_exact": 0, "front_exact": 0,
                         "unscored": 0, "ok_wall_seconds": [],
                         **{status: 0 for status in STATUSES}}
                  for name in methods}
    cells = defaultdict(dict)
    for run in planned:
        name = str(run["method"])
        stats = per_method[name]
        stats["planned"] += 1
        record = latest.get(str(run["run_id"]))
        status = str(record["status"]) if record is not None else "incomplete"
        stats[status] += 1
        success = None
        if status == "ok":
            rows = record.get("rows", [])
            nested = rows[0] if len(rows) == 1 and isinstance(rows[0], dict) else {}
            met = nested.get("metrics") if isinstance(nested.get("metrics"), dict) else {}
            # ``is True`` is intentional: missing scoring cannot be silently
            # imputed as either a measured success or a valid paired failure.
            if isinstance(met.get("success_selected"), bool):
                success = met["success_selected"] is True
                stats["selected_exact"] += int(success)
            else:
                stats["unscored"] += 1
            stats["front_exact"] += int(met.get("success_front") is True)
            wall = nested.get("total_seconds")
            if isinstance(wall, (float, int)):
                stats["ok_wall_seconds"].append(float(wall))
        elif status in {"crash", "timeout"}:
            success = False  # Never drop algorithm failure from success rate.
            if record.get("failure_kind") == "infrastructure" and (
                str(run["run_id"]) in ledger["remaining"]):
                success = None  # Retriable infra, unfinished observation.
        key = (str(run["case"]), int(run["data_seed"]), int(run["optimizer_seed"]))
        cells[key][name] = (status, success)

    for stats in per_method.values():
        values = stats.pop("ok_wall_seconds")
        stats["mean_wall_seconds_among_ok"] = (sum(values) / len(values) if values else None)
        stats["selected_exact_per_planned"] = (stats["selected_exact"] / stats["planned"]
                                               if stats["planned"] else None)
        assert sum(stats[status] for status in STATUSES) == stats["planned"]

    comparisons = {}
    for name in methods:
        if name == "default":
            continue
        a, b, groups = [], [], []
        unresolved = unsupported = 0
        for key in sorted(cells):
            row = cells[key]
            base = row.get("default")
            variant = row.get(name)
            if base is None or variant is None or (base[1] is None or variant[1] is None):
                if (base and base[0] == "unsupported") or (
                    variant and variant[0] == "unsupported"):
                    unsupported += 1
                else:
                    unresolved += 1
                continue
            a.append(bool(base[1]))
            b.append(bool(variant[1]))
            groups.append(key[0])
        contrast = {"n_pairs_expected": len(cells), "n_pairs_scored": len(a),
                    "unresolved_pairs": unresolved, "unsupported_pairs": unsupported,
                    "inference": "BLOCKED"}
        if not unresolved and not unsupported and len(set(groups)) >= 3:
            contrast.update(paired_effect_ci(a, b, groups, n_boot=2000, seed=101))
            contrast["inference"] = "RECORDED_DESCRIPTIVE_WITH_CI"
        comparisons[name] = contrast

    # All prespecified methods remain in the Holm family, even if blocked.
    names = [name for name in methods if name != "default"]
    corrected = holm_adjust([float(comparisons[name].get("cluster_signflip_p", 1.))
                             for name in names])
    for name, p_adj in zip(names, corrected):
        if "cluster_signflip_p" in comparisons[name]:
            comparisons[name]["holm_adjusted_p"] = p_adj

    return {"stage": "S1", "manifest_sha": manifest["manifest_sha"],
            "code_sha": manifest["code_sha"], "config_sha": manifest["config_sha"],
            "split_sha": manifest["split_sha"], "planned": len(planned),
            "ledger": ledger, "methods": per_method, "comparisons": comparisons,
            "inference_gate": ("BLOCKED" if any(c["inference"] == "BLOCKED"
                                               for c in comparisons.values())
                               else "PROVISIONAL_S1_ONLY")}


def write_s1_report(output_dir: Path, analysis: Mapping[str, object]) -> list[Path]:
    """Write raw machine-readable S1 analysis and restrained review Markdown."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "S1_SUMMARY.json"
    markdown_path = output_dir / "S1_SUMMARY.md"
    json_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2,
                                    sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# NIR-1 S1 — auditable full-search comparison", "",
             f"Frozen manifest: `{analysis['manifest_sha']}`; source SHA: `{analysis['code_sha']}`.",
             f"Planned full EPDE runs: **{analysis['planned']}**. "
             f"Inference gate: **{analysis['inference_gate']}**.", "",
             "Every planned ID is counted; a timeout, algorithm failure, or "
             "missing record is never classified as an exact recovery.", "",
             "## Methods, all planned identities", "",
             "| Method | Planned | OK | Crash | Timeout | Unsupported | Incomplete | "
             "PIC selected exact / planned | Mean wall(s), OK only |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, row in analysis["methods"].items():
        wall = row["mean_wall_seconds_among_ok"]
        lines.append(f"| `{name}` | {row['planned']} | {row['ok']} | {row['crash']} | "
                     f"{row['timeout']} | {row['unsupported']} | {row['incomplete']} | "
                     f"{row['selected_exact']}/{row['planned']} | "
                     f"{f'{wall:.1f}' if wall is not None else '—'} |")
    lines.extend(["", "## Paired vs PIC baseline (method minus baseline)", "",
                  "| Method | Scored / expected | Unresolved | Unsupported pairs | "
                  "Difference (pp) | 95% system-cluster CI (pp) | Holm system-level p | Gate |",
                  "|---|---:|---:|---:|---:|---|---:|---|"])
    for name, row in analysis["comparisons"].items():
        delta = (f"{row['delta_pp']:+.1f}" if "delta_pp" in row else "—")
        ci = (f"[{row['ci_low_pp']:+.1f}, {row['ci_high_pp']:+.1f}]"
              if "ci_low_pp" in row else "—")
        pval = (f"{row['holm_adjusted_p']:.3g}"
                if "holm_adjusted_p" in row else "—")
        lines.append(f"| `{name}` | {row['n_pairs_scored']}/{row['n_pairs_expected']} | "
                     f"{row['unresolved_pairs']} | {row['unsupported_pairs']} | "
                     f"{delta} | {ci} | {pval} | {row['inference']} |")
    lines.extend(["", "## Integrity and limits", "",
                  f"- Status totals: `{analysis['ledger']['status_counts']}`.",
                  f"- Invalid or corrupted records: **{len(analysis['ledger']['invalid_records'])}** "
                  "(not silently dropped from planned denominators).",
                  "- The recovery measure is the frozen PIC truth-free compromise "
                  "selector, scored offline against known truth.",
                  "- A system-clustered interval and exact system-signflip test only appear when all planned "
                  "pairs are verifiable and at least 3 systems are present.",
                  "- `PROVISIONAL_S1_ONLY` does not imply statistical significance, "
                  "transfer to real noisy data or completion of S2/S3 heldout validation.", ""])
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return [json_path, markdown_path]
