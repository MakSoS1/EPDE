"""Small, pure-CPU study CLI: no EPDE globals imported for S0."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import tempfile
from pathlib import Path

import yaml

from .audit import METHODS, evaluate_candidate_set
from .campaign import execute_shard, plan_campaign, restore_shard_records, resume_plan
from .identity import hash_file
from .reporting import render_review, render_s0_figures, summarize_s0_rows


def parse_seeds(text: str) -> list[int]:
    if "-" in text:
        start, end = map(int, text.split("-", 1))
        if end < start:
            raise ValueError("seed range must be increasing")
        return list(range(start, end + 1))
    return [int(v) for v in text.split(",")]


def atomic_jsonl(output: Path, rows: list[dict[str, object]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temp: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False,
                                         dir=output.parent, prefix=".nir1-") as stream:
            temp = stream.name
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False,
                                        allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, output)
    finally:
        if temp and os.path.exists(temp):
            os.unlink(temp)


def atomic_json(output: Path, body: dict[str, object]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False,
                                         dir=output.parent, prefix=".nir1-json-") as stream:
            temporary = stream.name
            json.dump(body, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EPDE NIR-1 auditable experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="S0 fixed candidate ranking, no evolution")
    audit.add_argument("--cases", default="T1,T2,T3,T4,T5,T6,T7,T8")
    audit.add_argument("--seeds", default="0-4")
    audit.add_argument("--methods", default=",".join(METHODS))
    audit.add_argument("--output", type=Path, required=True)
    plan = commands.add_parser("plan", help="Freeze one S0 Actions campaign from YAML")
    plan.add_argument("--launch", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    shard = commands.add_parser("shard", help="Run and checkpoint one fixed shard")
    shard.add_argument("--manifest", type=Path, required=True)
    shard.add_argument("--id", required=True)
    shard.add_argument("--output", type=Path, required=True)
    restore = commands.add_parser("restore", help="Verify and recover previous immutable shard artifacts")
    restore.add_argument("--manifest", type=Path, required=True)
    restore.add_argument("--id", required=True)
    restore.add_argument("--from", dest="source", type=Path, required=True)
    restore.add_argument("--output", type=Path, required=True)
    summary = commands.add_parser("summarize", help="Reconcile immutable ledger and records")
    summary.add_argument("--manifest", type=Path, required=True)
    summary.add_argument("--records", type=Path, required=True)
    summary.add_argument("--output", type=Path, required=True)
    report = commands.add_parser("report", help="Build honest S0 reviewer package")
    report.add_argument("--s0-results", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    options = parser.parse_args(argv)
    if options.command == "plan":
        config = yaml.safe_load(options.launch.read_text(encoding="utf-8"))
        if not isinstance(config, dict) or config.get("stage") != "S0":
            parser.error("Only the S0 fixed-candidate campaign is activated; full EPDE stages await S1 evidence")
        if int(config.get("shards", -1)) != 4:
            parser.error("The current GitHub Actions matrix has exactly four shards")
        from .audit import _repo_revision
        split_path = Path(__file__).resolve().parents[1] / "configs/splits.yaml"
        frozen = {**config, "code_sha": _repo_revision(),
                  "config_sha": hash_file(options.launch),
                  "split_sha": hash_file(split_path)}
        manifest = plan_campaign(frozen)
        atomic_json(options.output, manifest)
        print(json.dumps({"planned": len(manifest["runs"]),
                          "manifest_sha": manifest["manifest_sha"]}))
        return 0
    if options.command == "shard":
        return execute_shard(options.manifest, options.id, options.output)
    if options.command == "restore":
        manifest = json.loads(options.manifest.read_text(encoding="utf-8"))
        restored = restore_shard_records(manifest, options.id, options.source, options.output)
        print(json.dumps({"restored": restored, "shard": options.id}))
        return 0
    if options.command == "summarize":
        records = (sorted(options.records.rglob("*.json")) if options.records.exists()
                   else [])
        records = [p for p in records if len(p.stem) == 64 and
                   all(ch in "0123456789abcdef" for ch in p.stem)]
        body = resume_plan(options.manifest, records)
        atomic_json(options.output, body)
        print(json.dumps({"planned": body["planned"], "status_counts": body["status_counts"]}))
        return 0
    if options.command == "report":
        open_rows = gzip.open if options.s0_results.suffix == ".gz" else open
        with open_rows(options.s0_results, "rt", encoding="utf-8") as source:
            records = [json.loads(line) for line in source if line.strip()]
        result = summarize_s0_rows(records)
        written = render_review(result, options.output)
        written.extend(render_s0_figures(records, options.output / "figures"))
        print(json.dumps({"files": len(written), "method_trials": result["planned"],
                          "full_epde": result["full_epde_status"]}))
        return 0
    rows: list[dict[str, object]] = []
    for case in options.cases.split(","):
        for seed in parse_seeds(options.seeds):
            rows.extend(evaluate_candidate_set(case.strip(), options.methods.split(","), seed))
    atomic_jsonl(options.output, rows)
    print(json.dumps({"written": len(rows), "output": str(options.output),
                      "cases": options.cases, "seeds": options.seeds}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
