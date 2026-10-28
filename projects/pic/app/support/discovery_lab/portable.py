"""Versioned, data-aware handoff between the Streamlit Lab and native runs.

The ZIP contains *declarative* input only, never executable code. Download the
repository separately. No huge registered dataset is read on the web host.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
import zipfile
from pathlib import Path

SCHEMA = "epde-portable/1"
HEX = re.compile(r"^[0-9a-f]{64}$")
MAX_IMPORT = 2_000_000


def repository_source():
    """An immutable archive of the running checkout, never a moving branch.

    Git-less deployments may supply their build SHA explicitly. A dirty local
    checkout has no matching GitHub archive and must not advertise one.
    """
    from epde_bench.paths import REPO_ROOT
    repository = os.environ.get("EPDE_REPOSITORY_URL", "https://github.com/MakSoS1/EPDE").rstrip("/")
    if not re.fullmatch(r"https://github\.com/[\w.-]+/[\w.-]+", repository):
        raise ValueError("Invalid EPDE_REPOSITORY_URL")
    revision = os.environ.get("EPDE_REPOSITORY_REVISION", "")
    if (REPO_ROOT / ".git").exists():
        try:
            def git(*args):
                return subprocess.run(["git", "-C", str(REPO_ROOT), *args], check=True,
                                      capture_output=True, text=True, timeout=10).stdout.rstrip("\r\n")
            head = git("rev-parse", "HEAD")
            status = git("-c", "core.quotePath=true", "status", "--porcelain=v1", "--untracked-files=normal")
            if status:
                # Report only bounded, repository-relative change names. Never
                # read changed files or disclose their contents in the UI.
                details = status[:1200] + ("\n(additional paths omitted)" if len(status) > 1200 else "")
                raise ValueError("Repository export requires a clean committed checkout. "
                                 "Changed paths:\n" + details)
        except (OSError, subprocess.SubprocessError) as exc:
            raise ValueError("Could not identify the running repository revision") from exc
        if revision and revision != head:
            raise ValueError("EPDE_REPOSITORY_REVISION differs from the running checkout")
        revision = head
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Set EPDE_REPOSITORY_REVISION to the deployed commit in Git-less builds")
    return {"url": repository, "revision": revision,
            "archive_url": f"{repository}/archive/{revision}.zip"}


def digest(value):
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")).hexdigest()


def input_key(record):
    """A method-independent identity of the *numerical input*, not its optimizer.

    Include original data bytes, scientific source, loader options and the noise
    realization. Two identically named datasets are not necessarily identical.
    """
    identity = record.get("identity") or {}
    cfg = record.get("config") or {}
    data = identity.get("data_sha256")
    source = identity.get("source_sha256")
    if not isinstance(data, str) or not HEX.fullmatch(data):
        return None
    if not isinstance(source, str) or not HEX.fullmatch(source):
        return None
    if record.get("dataset") != (identity.get("job") or {}).get("dataset"):
        return None
    noise, seed = record.get("noise"), record.get("seed")
    if type(seed) is not int or type(noise) not in (int, float):
        return None
    if float(noise) != float(identity["job"].get("noise")) or seed != identity["job"].get("seed"):
        return None
    return digest({"schema": SCHEMA, "dataset": record["dataset"],
                   "data_sha256": data, "scientific_source_sha256": source,
                   "loader": cfg.get("loader") or {},
                   "noise": float(noise), "seed": seed,
                   "data_seed": cfg.get("nir1_data_seed", seed),
                   "noise_protocol": "problem.noisy/v1"})


def make_manifest(dataset, noise, seed, methods, *, repository=None):
    """Build a portable experiment without accessing its scientific arrays."""
    from epde_bench.config import load_config
    from epde_bench.datasets import REGISTRY
    from epde_bench.identity import _source_hash
    if dataset not in REGISTRY:
        raise ValueError("Unknown built-in equation")
    if type(seed) is not int or not 0 <= seed <= 999:
        raise ValueError("Seed must be within 0..999")
    if type(noise) not in (float, int) or not 0 <= noise <= 10:
        raise ValueError("Noise must be within 0..10")
    if not isinstance(methods, list) or not 1 <= len(methods) <= 4:
        raise ValueError("Choose one to four methods")
    items = []
    for entry in methods:
        if not isinstance(entry, dict) or entry.get("method") not in ("epde", "pdecomp", "deepmod", "discover"):
            raise ValueError("Unsupported portable algorithm")
        method = entry["method"]
        if method in [x["method"] for x in items]:
            raise ValueError("Duplicate method")
        variant = entry.get("variant") or ("pysindy_pdecomp_fixed" if method == "pdecomp" else f"{method}_pdecomp_native" if method in ("deepmod", "discover") else "default")
        overrides = entry.get("overrides") or {}
        if not isinstance(overrides, dict) or (set(overrides) -
                ({method} if method in ("pdecomp", "deepmod", "discover") else {"search"})):
            raise ValueError("Invalid scientific settings")
        cfg = load_config(dataset, variant, overrides=overrides)
        if cfg["method"] != method:
            raise ValueError("Configuration and method disagree")
        items.append({"method": method, "variant": variant, "overrides": overrides,
                      "loader": cfg.get("loader", {})})
    if len({digest(x["loader"]) for x in items}) != 1:
        raise ValueError("Methods have different loader inputs; they cannot be paired")
    manifest = {
        "schema": SCHEMA, "dataset": dataset, "noise": float(noise), "seed": seed,
        "scientific_source_sha256": _source_hash(), "runs": items,
        "origin": "EPDE Explorer / Run a search",
    }
    if repository is not None:
        manifest["repository"] = dict(repository)
    manifest["sha256"] = digest(manifest)
    return manifest


def bundle_bytes(manifest):
    if manifest.get("sha256") != digest({k:v for k,v in manifest.items() if k != "sha256"}):
        raise ValueError("Corrupt experiment manifest")
    payload = json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False)
    if len(payload.encode()) > 80_000:
        raise ValueError("Experiment settings too large")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("experiment.json", payload)
        z.writestr("README.txt",
            "EPDE Explorer — reproducible native experiment\n\n"
            "1. Download the matching EPDE repository ZIP from the demo.\n"
            + ("   " + manifest["repository"]["archive_url"] + "\n"
               if manifest.get("repository") else "") +
            "2. Extract it and place this file as experiment.zip in its root.\n"
            "3. Windows: double-click run-portable.cmd; macOS/Linux: bash run-portable.sh.\n"
            "4. Upload experiment-results.zip on the website's Results page.\n\n"
            "The launcher installs original pinned scientific dependencies. "
            "It does not send your data to the website.\n")
    return buffer.getvalue()


def read_manifest(bundle):
    if not isinstance(bundle, (str, Path)):
        raise ValueError("Expected a ZIP file path")
    with zipfile.ZipFile(bundle) as z:
        if z.namelist().count("experiment.json") != 1:
            raise ValueError("Missing or duplicated experiment.json")
        item = z.getinfo("experiment.json")
        if item.file_size > 80_000:
            raise ValueError("Manifest exceeds size limit")
        obj = json.loads(z.read(item))
    if not isinstance(obj, dict) or obj.get("schema") != SCHEMA:
        raise ValueError("Unknown experiment schema")
    original = {k:v for k,v in obj.items() if k != "sha256"}
    if obj.get("sha256") != digest(original):
        raise ValueError("Manifest checksum mismatch")
    return obj


def read_results(raw):
    """Read a bounded, untrusted result archive; never extract archive members."""
    from support.discovery_lab.contract import normalize
    if not isinstance(raw, bytes) or len(raw) > MAX_IMPORT:
        raise ValueError("Result archive must be at most 2 MB")
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        files = [x for x in z.infolist() if x.filename.startswith("results/")
                 and x.filename.endswith(".json") and "/" not in x.filename[8:]]
        if not 1 <= len(files) <= 4:
            raise ValueError("Expected one to four result JSON files")
        if sum(x.file_size for x in files) > MAX_IMPORT:
            raise ValueError("Uncompressed results exceed size limit")
        records = []
        for item in files:
            record = json.loads(z.read(item))
            if not isinstance(record, dict) or record.get("method") not in ("epde", "pdecomp", "deepmod", "discover"):
                raise ValueError("Unknown result method")
            if record.get("execution_location") != "user_computer" or record.get("portable_schema") != SCHEMA:
                raise ValueError("Not a portable native result")
            ident = record.get("identity") or {}
            original = {k:v for k,v in ident.items() if k != "sha256"}
            if ident.get("sha256") != digest(original):
                raise ValueError("Invalid original experiment identity")
            if record.get("config") != ident.get("config"):
                raise ValueError("Result settings differ from scientific run identity")
            if record.get("variant") != (ident.get("job") or {}).get("variant"):
                raise ValueError("Result variant differs from scientific run identity")
            configured_method = (ident.get("config") or {}).get("method")
            if configured_method and configured_method != record["method"]:
                raise ValueError("Result method differs from scientific run identity")
            key = input_key(record)
            if key is None or key != record.get("comparison_input_sha256"):
                raise ValueError("Input fingerprint mismatch")
            if not isinstance(record.get("portable_manifest_sha256"), str) or not HEX.fullmatch(record["portable_manifest_sha256"]):
                raise ValueError("Missing originating experiment checksum")
            records.append(normalize(record, record["method"]))
    if len({r["comparison_input_sha256"] for r in records}) != 1:
        raise ValueError("Result files do not share identical numerical input")
    if len({r["portable_manifest_sha256"] for r in records}) != 1:
        raise ValueError("Result files came from different experiment packages")
    if len({r["method"] for r in records}) != len(records):
        raise ValueError("Duplicate methods in result package")
    return records
