"""Scientific defaults for the unified Lab are record-specific and explicit.

The tests run against the actual YAML configuration loader, avoiding hand-written
pseudo-defaults. They do not claim any profile is universally optimal.
"""
from __future__ import annotations

import ast
import copy
from pathlib import Path
import unittest

from epde_bench.config import load_config, resolve_for_problem
from epde_bench.datasets import load
from epde_bench.paths import CONFIG_DIR
from support.discovery_lab.lab_settings import (
    EPDE_VARIANTS, _custom_base, validate_public_search,
)
from epde_bench.config import deep_merge


class LabScientificPresets(unittest.TestCase):
    def test_builtin_uses_original_layered_dataset_presets(self):
        for dataset in ("ode", "burgers", "vdp"):
            with self.subTest(dataset=dataset):
                cfg = load_config(dataset, "default")
                self.assertEqual(cfg["dataset"], dataset)
                problem = load(dataset, **cfg.get("loader", {}))
                resolved = resolve_for_problem(cfg, problem)
                self.assertIn("search_space", resolved)
                self.assertIn("preprocessing", resolved)
                self.assertGreater(resolved["evolution"]["population_size"], 0)
                self.assertGreater(resolved["evolution"]["training_epochs"], 0)
                self.assertEqual(cfg["search"]["solver"]["use_solver"], False)

    def test_custom_universal_profile_is_not_an_optimality_claim(self):
        ode = _custom_base(0, generated=True)
        pde = _custom_base(1, generated=False)
        self.assertEqual(ode["preprocessing"]["max_deriv_order"], [1])
        self.assertEqual(pde["preprocessing"]["max_deriv_order"], [2, 3])
        self.assertEqual(ode["evolution"]["population_size"], 12)
        self.assertEqual(ode["evolution"]["training_epochs"], 4)
        self.assertFalse(ode["solver"]["use_solver"])

    def test_public_settings_do_not_allow_unbounded_compute(self):
        cfg = _custom_base(0, generated=True)
        validate_public_search(cfg, (301,))
        cases = [
            ("too many epochs", {"evolution": {"training_epochs": 500}}),
            ("huge population", {"evolution": {"population_size": 500}}),
            ("arbitrary token", {"search_space": {"tokens": [{"family": "python_code"}]}}),
            ("run solver", {"solver": {"use_solver": True}}),
            ("huge cache", {"runtime": {"memory_for_cache": 100}}),
            ("custom operators", {"evolution": {"operators": {"X": {"command": "x"}}}}),
        ]
        for name, patch in cases:
            with self.subTest(name=name):
                with self.assertRaises((ValueError, KeyError, TypeError)):
                    validate_public_search(deep_merge(cfg, patch), (301,))

    def test_trusted_local_computation_has_larger_but_validated_budgets(self):
        cfg = _custom_base(0, generated=True)
        bigger = deep_merge(cfg, {"evolution": {
            "population_size": 128, "training_epochs": 75},
            "runtime": {"memory_for_cache": 8}})
        with self.assertRaises(ValueError):
            validate_public_search(bigger, (301,))
        validate_public_search(bigger, (301,), override=True)
        bad = deep_merge(bigger, {"evolution": {"population_size": 600}})
        with self.assertRaises(ValueError):
            validate_public_search(bad, (301,), override=True)

    def test_epde_variants_are_method_separated(self):
        self.assertIn("default", EPDE_VARIANTS)
        self.assertIn("poly", EPDE_VARIANTS)
        self.assertNotIn("pysindy", EPDE_VARIANTS)

    def test_browser_configuration_reaches_worker_as_override(self):
        app = Path(__file__).resolve().parents[1] / "app"
        lab = (app / "views/10_Discovery_lab.py").read_text(encoding="utf-8")
        worker = (app / "support/discovery_lab/worker.py").read_text(encoding="utf-8")
        ast.parse(lab)
        ast.parse(worker)
        self.assertIn('"overrides": overrides if m != "llm" else {}', lab)
        self.assertIn('if "overrides" in req:', worker)
        self.assertIn("validate_public_search(resolve_for_problem(cfg, problem), problem.shape, override=not hosted)", worker)
        self.assertIn("validate_public_search(search_cfg, problem.shape, override=not hosted)", worker)

    def test_advanced_fields_and_yaml_transport_exist(self):
        app = Path(__file__).resolve().parents[1] / "app"
        ui = (app / "support/discovery_lab/lab_settings.py").read_text(encoding="utf-8")
        for label in (
            "Maximum field power", "Maximum derivative power",
            "Sparsity operator", "Instability estimator", "Second objective",
            "Maximum derivative order on", "Polynomial window",
            "Import search-only YAML overrides", "Export effective YAML",
            "Specify boundary trimming explicitly", "Allow products of 2 factors",
        ):
            with self.subTest(label=label):
                self.assertIn(label, ui)


if __name__ == "__main__":
    unittest.main()
