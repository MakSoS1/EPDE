"""Scientific and security regression tests for browser-to-native handoff."""
from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

PIC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PIC), str(PIC / "app")]

from support.discovery_lab.portable import (
    SCHEMA, bundle_bytes, digest, input_key, make_manifest, read_manifest, read_results,
)
from portable_experiment import prepare


def mock_record(method="epde", data="a" * 64):
    identity = {"schema": 1, "job": {"dataset": "ode", "variant": "default",
                                    "noise": 0.0, "seed": 0},
                "config": {"loader": {}}, "source_sha256": "b" * 64,
                "data_sha256": data, "dependencies": {}}
    identity["sha256"] = digest(identity)
    result = {"dataset": "ode", "method": method, "variant": "default",
              "noise": 0.0, "seed": 0, "identity": identity,
              "config": {"loader": {}}, "status": "unsupported",
              "reason": "test synthetic unsupported", "execution_location": "user_computer",
              "portable_schema": SCHEMA, "portable_manifest_sha256": "c" * 64}
    result["comparison_input_sha256"] = input_key(result)
    return result


def archive(records):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        for result in records:
            z.writestr("results/" + result["method"] + ".json", json.dumps(result))
    return buffer.getvalue()


class HandoffTests(unittest.TestCase):
    def test_experiment_zip_roundtrip_without_dataset_materialization(self):
        manifest = make_manifest("ns", 0.0, 7, [
            {"method": "epde", "variant": "default",
             "overrides": {"search": {"evolution": {"population_size": 8,
                                                     "training_epochs": 3}}}}
        ])
        self.assertEqual(manifest["dataset"], "ns")
        self.assertEqual(manifest["seed"], 7)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "experiment.zip"
            path.write_bytes(bundle_bytes(manifest))
            self.assertEqual(read_manifest(path), manifest)
            self.assertEqual(prepare(path), manifest)

    def test_exported_repository_manifest_reaches_native_launcher(self):
        revision = "a" * 40
        repository = {"url": "https://github.com/emil817/EPDE_demo", "revision": revision,
                      "archive_url": f"https://github.com/emil817/EPDE_demo/archive/{revision}.zip"}
        manifest = make_manifest("burgers", 1.0, 7, [
            {"method": "epde", "variant": "default"},
            {"method": "pdecomp", "variant": "pysindy_pdecomp_native"},
            {"method": "deepmod", "variant": "deepmod_pdecomp_native",
             "overrides": {"deepmod": {"overrides": {"max_iterations": 2}}}},
            {"method": "discover", "variant": "discover_pdecomp_native",
             "overrides": {"discover": {"overrides": {"n_samples": 20, "batch_size": 10}}}},
        ], repository=repository)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "experiment.zip"
            path.write_bytes(bundle_bytes(manifest))
            actual = prepare(path)
        self.assertEqual(actual, manifest)
        self.assertEqual(actual["repository"], repository)
        self.assertEqual(actual["sha256"], manifest["sha256"])

    def test_repository_metadata_does_not_bypass_source_validation(self):
        manifest = make_manifest("burgers", 0, 0, [{"method": "epde", "variant": "default"}],
                                 repository={"url": "https://github.com/emil817/EPDE_demo",
                                             "revision": "a" * 40,
                                             "archive_url": "https://github.com/emil817/EPDE_demo/archive/" + "a" * 40 + ".zip"})
        # A correctly checksummed ZIP from different scientific sources must
        # still fail, even though repository metadata is retained unchanged.
        manifest["scientific_source_sha256"] = "0" * 64
        manifest["sha256"] = digest({k: v for k, v in manifest.items() if k != "sha256"})
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "experiment.zip"
            path.write_bytes(bundle_bytes(manifest))
            with self.assertRaisesRegex(ValueError, "scientific sources/configuration differ"):
                prepare(path)

    def test_fingerprint_is_method_neutral(self):
        a, b = mock_record("epde"), mock_record("pdecomp")
        self.assertEqual(input_key(a), input_key(b))
        self.assertEqual(len(read_results(archive([a, b]))), 2)

    def test_different_data_not_comparable(self):
        a, b = mock_record("epde"), mock_record("pdecomp", "d" * 64)
        self.assertNotEqual(input_key(a), input_key(b))
        with self.assertRaisesRegex(ValueError, "identical numerical input"):
            read_results(archive([a, b]))

    def test_manifest_or_identity_tampering_is_rejected(self):
        result = mock_record()
        result["identity"]["source_sha256"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "identity"):
            read_results(archive([result]))

    def test_result_without_provenance_is_rejected(self):
        result = mock_record()
        result.pop("portable_manifest_sha256")
        with self.assertRaisesRegex(ValueError, "checksum"):
            read_results(archive([result]))

    def test_host_still_refuses_heavy(self):
        from support.discovery_lab.execution_policy import classify_epde
        allowed, _, _ = classify_epde(
            "ns", (40, 80, 100), 3,
            {"evolution": {"population_size": 4, "training_epochs": 1}})
        self.assertFalse(allowed)


if __name__ == "__main__":
    unittest.main()
