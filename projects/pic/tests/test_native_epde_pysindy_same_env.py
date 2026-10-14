"""Prove real EPDE + real PySINDy run in ONE locked native environment.

This does not establish that two different equation-discovery frameworks have
identical expressive power, numerical objectives, or supported PDE classes.
"""
from __future__ import annotations

import json
import os
import sys
import unittest

from epde_bench.config import load_config, variants
from epde_bench.datasets import load
from epde_bench.baselines import unsupported_reason
from epde_bench.runner import run_one


class NativeFrameworkCompatibility(unittest.TestCase):
    def test_same_environment_loads_epde_pysindy_and_torch(self):
        import epde
        import pysindy
        import torch
        import scipy
        import numpy
        self.assertTrue(hasattr(epde, "EpdeSearch"))
        self.assertTrue(hasattr(pysindy, "STLSQ"))
        self.assertTrue(hasattr(torch, "Tensor"))
        self.assertTrue(numpy.__version__)
        self.assertTrue(scipy.__version__)
        self.assertIn(load_config("ode", "pysindy")["method"], ("pysindy",))
        self.assertEqual(load_config("ode", "default")["method"], "epde")
        print("VERSIONS", json.dumps({
            "python": sys.version.split()[0],
            "numpy":numpy.__version__, "scipy":scipy.__version__,
            "pysindy":pysindy.__version__, "torch":torch.__version__,
        }), flush=True)

    def test_pysindy_representability_is_not_epde_representability(self):
        self.assertIsNotNone(unsupported_reason("pde_divide"))
        self.assertIsNotNone(unsupported_reason("ns"))
        for name in ("ode", "vdp", "burgers", "kdv", "heat"):
            with self.subTest(dataset=name):
                self.assertIsNone(unsupported_reason(name))
        self.assertTrue(len(variants()) >= 2)

    def test_actual_pysindy_on_original_second_order_forced_oscillator(self):
        record = run_one("ode", variant="pysindy", noise=0.0, seed=0)
        print("PYSINDY", json.dumps({
            "status":record["status"],
            "equations":record.get("selected"),
            "error":str(record.get("error",""))[:240],
        }), flush=True)
        self.assertEqual(record["status"], "ok", record.get("error"))
        self.assertTrue(record.get("selected"))
        self.assertIn("d^2", " ".join(record["selected"]))

    def test_real_epde_with_original_engine_not_educational_substitute(self):
        # Limited scientific budget to bound CI, but the ORIGINAL epde.EpdeSearch
        # is called by epde_bench.runner.run_one, NOT educational EvoSparse.
        record = run_one("ode", variant="default", noise=0.0, seed=0,
            overrides={"search": {
                "evolution": {"population_size": 4, "training_epochs": 1},
                "runtime": {"memory_for_cache": 2,
                            "verbose_params":{"show_iter_idx":False,
                                              "candidate_objectives":False}}
            }})
        print("EPDE", json.dumps({
            "status":record["status"],
            "equations":record.get("selected"),
            "error":str(record.get("error",""))[:240],
        }), flush=True)
        self.assertEqual(record["method"], "epde")
        self.assertEqual(record["status"], "ok", record.get("error"))
        self.assertTrue(record.get("selected"))


if __name__ == "__main__":
    unittest.main()
