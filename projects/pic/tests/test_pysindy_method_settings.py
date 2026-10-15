"""Native PySINDy settings and separate EPDE/PySINDy configuration regression.

Unlike UI-only tests, the integrated cases execute real original scientific
adapters. These experiments are deliberately low budget, not claims of optimum.
"""
from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from epde_bench.baselines import _columns, _n_columns, run_pysindy
from epde_bench.config import load_config
from epde_bench.datasets import load
from support.discovery_lab import job_queue as queue, worker
from support.discovery_lab.sindy_settings import validate_sindy_options
from support.discovery_lab.ode import generate


class SindySpecificTests(unittest.TestCase):
    def test_legacy_auto_defaults_remain_bitwise_science_equivalent(self):
        problem = load("ode")
        data = problem.noisy(0, 0)
        cfg = load_config("ode", "pysindy")
        baseline = run_pysindy(problem, data, cfg, seed=0)
        explicit = copy.deepcopy(cfg)
        explicit["pysindy"].update(
            threshold="auto", normalize_columns="auto",
            max_terms_product=2, validation_fraction=0.20)
        repeated = run_pysindy(problem, data, explicit, seed=0)
        self.assertEqual(baseline[0], repeated[0])
        self.assertEqual(baseline[1], repeated[1])
        self.assertEqual(baseline[3]["lhs"], repeated[3]["lhs"])

    def test_native_stlsq_options_change_actual_executed_settings(self):
        problem = load("ode")
        data = problem.noisy(0, 0)
        cfg = load_config("ode", "pysindy")
        cfg["pysindy"].update({
            "threshold": 0.0001,
            "normalize_columns": "on",
            "max_terms_product": 1,
            "validation_fraction": 0.30,
        })
        result = run_pysindy(problem, data, cfg, seed=0)
        self.assertEqual(len(result[0]), 1)
        info = result[3]
        self.assertEqual(info["threshold_mode"], 0.0001)
        self.assertEqual(info["normalization_mode"], "on")
        self.assertEqual(info["validation_fraction"], 0.30)
        self.assertEqual(info["max_terms_product"], 1)
        self.assertTrue(all(v is True for v in info["normalize_columns"].values()))

    def test_disabled_products_really_remove_cross_feature_columns(self):
        blocks = [
            ("u", np.array([1., 2., 3.]), "field", True, 0),
            ("ux", np.array([4., 5., 6.]), "derivative", True, 0),
        ]
        singles = _columns(blocks, 1, max_factors=1)
        products = _columns(blocks, 1, max_factors=2)
        self.assertEqual(len(singles), 2)
        self.assertEqual(len(products), 3)
        self.assertEqual(_n_columns(blocks, 1), 2)
        self.assertEqual(_n_columns(blocks, 2), 3)

    def test_reject_wrong_optimizer_options_before_job_runs(self):
        for invalid in [
            {"differentiation": "__import__('os')"},
            {"threshold": -1},
            {"threshold": float("nan")},
            {"max_terms_product": 3},
            {"validation_fraction": 0.99},
            {"normalize_columns": "not-real"},
            {"import_path": "__import__('os')"},
        ]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_sindy_options(invalid)
        with self.assertRaises(ValueError):
            validate_sindy_options({"differentiation": "finite_difference"},
                                   supplied_derivatives=True)

    def test_separate_upload_configs_and_matched_data(self):
        from epde_bench.config import load_config
        from support.discovery_lab.lab_settings import _custom_base
        with tempfile.TemporaryDirectory() as tmp:
            old_root, old_lock = queue.ROOT, queue.MANAGER_LOCK
            queue.ROOT = Path(tmp)
            queue.MANAGER_LOCK = queue.ROOT / "manager.lock"
            try:
                owner = queue.new_owner()
                arrays = generate("-0.5*u", 6., 1., 151)
                search_epde = _custom_base(0, generated=True)
                search_epde["evolution"]["population_size"] = 4
                search_epde["evolution"]["training_epochs"] = 1
                search_sindy = copy.deepcopy(search_epde)
                search_sindy["search_space"]["data_fun_pow"] = 1
                settings = {
                    "seed": 0, "noise": 0,
                    "search": search_epde, "origin": "generated_ode",
                    "method_configs": {
                        "epde": {"search": search_epde},
                        "pysindy": {
                            "search": search_sindy,
                            "pysindy": {
                                "differentiation": "epde",
                                "threshold": "auto",
                                "max_terms_product": 1,
                                "normalize_columns": "auto",
                                "validation_fraction": 0.20,
                            },
                        },
                    },
                }
                path, conf = queue.save_upload(owner, arrays, settings)
                r = {
                    "source": "upload", "dataset": "custom",
                    "data_path": str(path), "config_path": str(conf),
                    "noise": 0, "seed": 0,
                }
                epde = worker._run_upload({**r, "method": "epde"}, queue.job_path(owner, "a"*32),
                                          hosted=False)
                sindy = worker._run_upload({**r, "method": "pysindy"}, queue.job_path(owner, "b"*32),
                                           hosted=False)
                self.assertEqual(epde["method"], "epde")
                self.assertEqual(sindy["method"], "pysindy")
                self.assertEqual(epde["status"], "ok", epde.get("error"))
                self.assertEqual(sindy["status"], "ok", sindy.get("error"))
                self.assertEqual(epde["search_config"]["search_space"]["data_fun_pow"], 3)
                self.assertEqual(sindy["search_config"]["search_space"]["data_fun_pow"], 1)
                self.assertEqual(sindy["baseline"]["max_terms_product"], 1)
                self.assertEqual(epde["data_sha256"], sindy["data_sha256"])
            finally:
                queue.ROOT, queue.MANAGER_LOCK = old_root, old_lock


if __name__ == "__main__":
    unittest.main()
