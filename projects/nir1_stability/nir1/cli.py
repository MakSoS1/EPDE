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
from .reporting import render_pilot_addendum, render_review, render_s0_figures, summarize_s0_rows


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
    epde = commands.add_parser("epde", help="One actual PIC full-search run, separately from S0")
    epde.add_argument("--dataset", required=True)
    epde.add_argument("--variant", default="default")
    epde.add_argument("--seed", type=int, default=0)
    epde.add_argument("--noise", type=float, default=0.)
    epde.add_argument("--data-seed", type=int)
    epde.add_argument("--smoke", action="store_true", help="Use one generation for a connectivity test; never research evidence")
    epde.add_argument("--overrides-json", default="{}", help="Frozen JSON search overrides (never truth labels)")
    epde.add_argument("--output", type=Path, required=True)
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
    analyze = commands.add_parser("analyze-s1", help="Failure-aware S1 method and paired analysis")
    analyze.add_argument("--manifest", type=Path, required=True)
    analyze.add_argument("--records", type=Path, required=True)
    analyze.add_argument("--output", type=Path, required=True)
    report = commands.add_parser("report", help="Build honest S0 reviewer package")
    report.add_argument("--s0-results", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--ledger-summary", type=Path)
    report.add_argument("--epde-runs", type=Path)
    report.add_argument("--smoke-runs", type=Path)
    report.add_argument("--actions-status", type=Path)
    options = parser.parse_args(argv)
    if options.command == "epde":
        from .epde_adapter import run_nir1_epde
        if options.smoke and options.overrides_json != "{}":
            parser.error("Smoke settings cannot be mixed with research budget overrides")
        overrides = (json.loads(options.overrides_json)
                     if not options.smoke else
                     {"search": {"evolution": {"population_size": 4, "training_epochs": 1}}})
        if not isinstance(overrides, dict):
            parser.error("Search overrides must be a JSON object")
        record = run_nir1_epde(options.dataset, options.variant, options.seed,
                               options.noise, data_seed=options.data_seed, overrides=overrides)
        record["nir1_smoke_only"] = bool(options.smoke)
        atomic_json(options.output, record)
        print(json.dumps({"status": record["status"], "full_search": True,
                          "smoke_only": bool(options.smoke), "output": str(options.output)}))
        return 0 if record["status"] == "ok" else 1
    if options.command == "plan":
        config = yaml.safe_load(options.launch.read_text(encoding="utf-8"))
        if not isinstance(config, dict) or config.get("stage") not in {"S0", "S1", "S2", "S3"}:
            parser.error("Launch manifest needs a known S0–S3 stage")
        if config["stage"] in {"S2", "S3"} and not config.get("frozen_confirmation_sha"):
            parser.error("Heldout confirmation requires a frozen S1 decision SHA")
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
    if options.command == "analyze-s1":
        from .s1_aggregate import analyze_s1, write_s1_report
        manifest = json.loads(options.manifest.read_text(encoding="utf-8"))
        if manifest["stage"] != "S1":
            print(json.dumps({"status": "NOT APPLICABLE", "stage": manifest["stage"]}))
            return 0
        files = (sorted(options.records.rglob("*.json"))
                 if options.records.exists() else [])
        files = [p for p in files if len(p.stem) == 64 and
                 all(char in "0123456789abcdef" for char in p.stem)]
        result = analyze_s1(manifest, files)
        write_s1_report(options.output, result)
        print(json.dumps({"planned": result["planned"],
                          "inference_gate": result["inference_gate"]}))
        return 0
    if options.command == "report":
        open_rows = gzip.open if options.s0_results.suffix == ".gz" else open
        with open_rows(options.s0_results, "rt", encoding="utf-8") as source:
            records = [json.loads(line) for line in source if line.strip()]
        result = summarize_s0_rows(records)
        full = ([json.loads(p.read_text(encoding="utf-8")) for p in
                 sorted(options.epde_runs.rglob("*.json"))
                 if "-precommit" not in p.stem]
                if options.epde_runs and options.epde_runs.exists() else [])
        if any(not record.get("nir1_smoke_only", False) for record in full):
            result["full_epde_status"] = "S1 PILOT ONLY"
        written = render_review(result, options.output)
        written.extend(render_s0_figures(records, options.output / "figures"))
        if options.ledger_summary or options.epde_runs or options.smoke_runs or options.actions_status:
            ledger = (json.loads(options.ledger_summary.read_text(encoding="utf-8"))
                      if options.ledger_summary else None)
            if options.smoke_runs and options.smoke_runs.exists():
                full.extend(json.loads(p.read_text(encoding="utf-8")) for p in
                            sorted(options.smoke_runs.rglob("*.json")))
            actions = (json.loads(options.actions_status.read_text(encoding="utf-8"))
                       if options.actions_status else [])
            written.extend(render_pilot_addendum(options.output, full, ledger,
                                                 actions_runs=actions))
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
