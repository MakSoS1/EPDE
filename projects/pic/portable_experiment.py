"""Execute a Streamlit-exported experiment using original scientific methods.

From repository root: bash run-portable.sh, or double-click run-portable.cmd.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

PIC_DIR = Path(__file__).resolve().parent
APP_DIR = PIC_DIR / "app"
REPO_ROOT = PIC_DIR.parents[1]
sys.path[:0] = [str(APP_DIR), str(PIC_DIR), str(REPO_ROOT)]


def prepare(bundle: Path):
    from support.discovery_lab.portable import make_manifest, read_manifest
    manifest = read_manifest(bundle)
    expected = make_manifest(manifest["dataset"], manifest["noise"],
                             manifest["seed"], manifest["runs"])
    if expected != manifest:
        raise ValueError(
            "Experiment and checked-out scientific sources/configuration differ. "
            "Use the matching repository revision; settings were not changed.")
    return manifest


def one(manifest, method, path):
    from epde_bench.runner import run_one
    from support.discovery_lab import contract
    from support.discovery_lab.portable import SCHEMA, input_key
    from support.discovery_lab.worker import evaluate, _sanitize, _finite_objectives

    choices = [r for r in manifest["runs"] if r["method"] == method]
    if len(choices) != 1:
        raise ValueError("Method not included in experiment")
    choice = choices[0]
    os.environ["EPDE_LOCAL_FULL_RUN"] = "1"
    record = run_one(manifest["dataset"], choice["variant"], manifest["noise"],
                     manifest["seed"], overrides=choice["overrides"])
    record = contract.normalize(_sanitize(_finite_objectives(record)), method)
    record["execution_location"] = "user_computer"
    record["portable_schema"] = SCHEMA
    record["portable_manifest_sha256"] = manifest["sha256"]
    record["comparison_input_sha256"] = input_key(record)
    if record["comparison_input_sha256"] is None:
        raise ValueError("No verifiable native input fingerprint")
    if record["status"] == "ok":
        try:
            record["evaluation"] = _sanitize(evaluate(record))
        except Exception as exc:
            record["evaluation"] = {"error": type(exc).__name__}
    payload = json.dumps(_sanitize(record), ensure_ascii=False, allow_nan=False).encode("utf-8")
    if len(payload) > 800_000:
        record.pop("evaluation", None)
        payload = json.dumps(_sanitize(record), ensure_ascii=False, allow_nan=False).encode("utf-8")
    if len(payload) > 800_000:
        raise ValueError("Native result too large for safe browser import")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    print(f"{method}: {record['status']}; result: {path}")
    return record["status"] in ("ok", "unsupported")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=REPO_ROOT / "experiment.zip")
    parser.add_argument("--method", choices=("epde", "pysindy"), default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    manifest = prepare(args.bundle.resolve())
    if args.method:
        if args.output is None:
            parser.error("--output required for a single method")
        return 0 if one(manifest, args.method, args.output.resolve()) else 1

    folder = REPO_ROOT / "results" / "portable" / manifest["sha256"][:16]
    folder.mkdir(parents=True, exist_ok=True)
    outputs, all_success = [], True
    for item in manifest["runs"]:
        method = item["method"]
        target = folder / (method + ".json")
        result = subprocess.run([
            sys.executable, str(Path(__file__).resolve()),
            "--bundle", str(args.bundle.resolve()),
            "--method", method, "--output", str(target),
        ], check=False)
        if result.returncode != 0 or not target.is_file():
            all_success = False
            print(f"{method} failed; see the error above", file=sys.stderr)
        else:
            outputs.append(target)
    if not outputs:
        return 1
    out = REPO_ROOT / "experiment-results.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for source in outputs:
            z.write(source, arcname="results/" + source.name)
    print(f"Upload {out} on the website's Results page.")
    return 0 if all_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
