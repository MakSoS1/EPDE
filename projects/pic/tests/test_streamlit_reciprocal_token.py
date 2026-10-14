"""Small real-EPDE smoke for newly added reciprocal coordinate token."""
import sys
import unittest
from pathlib import Path

import numpy as np

PIC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PIC), str(PIC.parent.parent)]


class ReciprocalToken(unittest.TestCase):
    def test_token_family_builds(self):
        from epde_bench.config import split_bench_tokens
        cfg = {"search_space": {"tokens": [
            {"family": "inverse_space_coordinate", "axis": 1,
             "dimensionality": 1}]}}
        stripped, families = split_bench_tokens(cfg)
        self.assertEqual(stripped["search_space"]["tokens"], [])
        self.assertEqual(len(families), 1)
        self.assertEqual(families[0].token_family.ftype, "inverse_space_coordinate")

    def test_reject_out_of_domain_axis(self):
        from epde_bench.tokens import inverse_space_coordinate
        with self.assertRaisesRegex(ValueError, "spatial axis"):
            inverse_space_coordinate(dimensionality=0, axis=1)

    def test_real_EPDE_search_with_nonzero_x(self):
        from epde_bench.problem import Problem, mesh
        from epde_bench.config import load_config, resolve_for_problem, deep_merge
        from epde_bench.runner import discover
        from epde_bench import env
        env.seed_everything(0)
        t = np.linspace(0.0, 0.3, 26)
        x = np.linspace(1.0, 2.0, 25)
        T, X = mesh(t, x)
        u = np.exp(-T) * (1 + 0.1 * np.sin(2 * np.pi * (X - 1)))
        problem = Problem(name="reciprocal_smoke", title="Reciprocal coordinate integration",
                          kind="pde_1d", source="synthetic",
                          grids=(T, X), data={"u": u}, axis_names=("t", "x"))
        cfg = load_config("nir1_heat")
        cfg = deep_merge(cfg, {"search": {
            "preprocessing": {"default_preprocessor_type": "FD", "max_deriv_order": [1, 2]},
            "search_space": {"tokens": [{"family": "inverse_space_coordinate", "axis": 1}]},
            "evolution": {"population_size": 4, "training_epochs": 1}}})
        search_cfg = resolve_for_problem(cfg, problem)
        search, texts, objectives, seconds = discover(problem, search_cfg)
        try:
            self.assertTrue(isinstance(texts, list))
            self.assertTrue(isinstance(objectives, list))
        finally:
            search.close()


if __name__ == "__main__":
    unittest.main()
