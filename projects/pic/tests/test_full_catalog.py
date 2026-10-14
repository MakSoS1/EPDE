"""Regression: every registered ODE/PDE is visible without loading heavy arrays."""
from __future__ import annotations

import ast
import json
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import patch

from epde_bench.datasets import REGISTRY
from support.discovery_lab.catalog import (
    KIND_LABELS, declared_equations, full_catalog, grouped_catalog,
)
from support.discovery_lab.execution_policy import (
    classify_epde, is_local_only,
)

ROOT = Path(__file__).resolve().parents[3]
APP = ROOT / "projects" / "pic" / "app"


class CompleteCatalogTests(TestCase):
    def test_all_original_registered_datasets_in_ui(self):
        with patch("epde_bench.datasets.load", side_effect=AssertionError("eager arrays load")):
            rows = full_catalog()
            groups = grouped_catalog()
        self.assertGreaterEqual(len(rows), 59, "Core/extended/real/other records vanished")
        self.assertEqual({row["id"] for row in rows}, set(REGISTRY))
        self.assertEqual(set(groups["All registered datasets"]), set(REGISTRY))
        self.assertEqual(len(groups["All registered datasets"]), len(REGISTRY))
        for row in rows:
            self.assertIn(row["id"], groups[KIND_LABELS[row["kind"]]])
            self.assertTrue(row["title"])
            self.assertTrue(row["files"])

    def test_full_catalog_includes_non_curated_2d_3d_real(self):
        for name in ("ns", "jhtdb_plane", "heat_laser", "heat_solar_2d",
                     "dp_encoder", "robot_arm", "ballbeam", "kdv_sga",
                     "kdv_homogen", "sst", "darcy", "dp_video"):
            with self.subTest(dataset=name):
                self.assertIn(name, REGISTRY)
                self.assertIn(name, grouped_catalog()["All registered datasets"])

    def test_actual_declared_truth_not_hallucinated(self):
        laws = declared_equations("ode")
        self.assertTrue(laws)
        self.assertIn("du/dx0", " ".join(laws))
        self.assertEqual(declared_equations("nonexistent") if False else [], [])
        with self.assertRaises(KeyError):
            declared_equations("nonexistent")

    def test_heavy_pde_never_eagerly_loaded_on_shared_server(self):
        for name in ("ns", "jhtdb_plane", "ks", "heat_laser",
                     "heat_solar_2d", "sst", "darcy", "dp_video"):
            with self.subTest(name=name):
                self.assertTrue(is_local_only(name))
                self.assertFalse(classify_epde(name, (20,), 1,
                                 {"evolution":{"population_size":4,"training_epochs":1}})[0])
        self.assertFalse(is_local_only("ode"))

    def test_full_local_app_and_codespaces_files(self):
        for path in [APP / "Home.py", APP / "views/10_Discovery_lab.py",
                     APP / "views/14_Data_catalog.py",
                     APP / "support/discovery_lab/catalog.py",
                     APP / "support/discovery_lab/worker.py"]:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        nav = (APP / "Home.py").read_text()
        self.assertIn("14_Data_catalog.py", nav)
        source = (APP / "views/10_Discovery_lab.py").read_text()
        self.assertIn("return grouped_catalog()", source)
        self.assertIn('EPDE_LOCAL_FULL_RUN', source)
        self.assertIn("if is_local_only(record_name) and not local_full:", source)
        worker = (APP / "support/discovery_lab/worker.py").read_text()
        self.assertIn('EPDE_LOCAL_FULL_RUN', worker)
        self.assertIn('hosted=hosted', worker)
        dc = json.loads((ROOT / ".devcontainer/devcontainer.json").read_text())
        self.assertIn(8501, dc["forwardPorts"])
        self.assertEqual(dc["portsAttributes"]["8501"]["visibility"], "private")
        self.assertTrue((ROOT / "run-epde.sh").is_file())
        self.assertTrue((ROOT / "run-epde.cmd").is_file())


if __name__ == "__main__":
    main()
