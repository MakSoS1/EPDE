"""Catalog-to-wizard transition: exercise the real page's event handler AST.

No Streamlit server or scientific dataset is required. Only the button branch
is executed, with the session mapping and page switch recorded. Full browser
navigation is tested separately in smoke_catalog_restart.mjs.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

APP = Path(__file__).resolve().parents[1] / "app"


def launch_from_catalog(state, selected="ns"):
    source = ast.parse((APP / "views/14_Data_catalog.py").read_text(encoding="utf-8"))
    handlers = [node for node in ast.walk(source)
                if isinstance(node, ast.If) and isinstance(node.test, ast.Call)
                and isinstance(node.test.func, ast.Attribute)
                and node.test.func.attr == "button" and node.test.args
                and isinstance(node.test.args[0], ast.Constant)
                and node.test.args[0].value == "Use this record in Lab"]
    if len(handlers) != 1:
        raise AssertionError("Expected one actual catalog launch button")
    calls = []
    namespace = {"st": SimpleNamespace(session_state=state, switch_page=calls.append),
                 "selected": selected}
    exec(compile(ast.Module(body=handlers[0].body, type_ignores=[]),
                 "catalog-launch", "exec"), namespace)
    return calls


class CatalogRestartTests(unittest.TestCase):
    def test_new_record_resets_stage_draft_and_approval(self):
        for stage in (1, 2, 3):
            with self.subTest(stage=stage):
                state = {
                    "lab_stage": stage,
                    "lab_draft_input": {"record_name": "ode"},
                    "lab_draft_config": {"old_ode_config": True},
                    "lab_confirm_science": True,
                    "lab_started_jobs": ["finished-ode-run"],
                    "lab_widgets_saved": {"lab_builtin_equation": "ode",
                                          "lab_confirm_science": True},
                    "lab_builtin_equation": "ode",
                    "lab_builtin_group": "Core benchmark",
                    "lab_input_source": "Upload your data",
                    "lab_deepmod_ode_native_max_iterations": 5000,
                    "lab_discover_ode_native_n_samples": 5000,
                    "portable_epde_population_ode": 24,
                    "heavy_noise_ode": 9,
                    "native_advanced_epde_ode": "old-yaml",
                    "discovery_owner": "session-to-preserve",
                    "discovery_selected_job": "finished-ode-run",
                    "epde_appearance": "Dark",
                    "catalog_page": 2,
                }
                self.assertEqual(launch_from_catalog(state), ["views/10_Discovery_lab.py"])
                self.assertEqual(state["catalog_requested_record"], "ns")
                self.assertEqual(state.get("lab_stage"), 1)
                self.assertEqual(state.get("lab_input_source"), "Built-in equation")
                self.assertNotIn("lab_draft_input", state)
                self.assertNotIn("lab_draft_config", state)
                self.assertFalse(state.get("lab_confirm_science", False))
                self.assertNotIn("lab_widgets_saved", state)
                self.assertNotIn("lab_started_jobs", state)
                self.assertNotIn("lab_builtin_equation", state)
                self.assertNotIn("lab_deepmod_ode_native_max_iterations", state)
                self.assertNotIn("lab_discover_ode_native_n_samples", state)
                self.assertNotIn("portable_epde_population_ode", state)
                self.assertNotIn("heavy_noise_ode", state)
                self.assertNotIn("native_advanced_epde_ode", state)
                self.assertEqual(state["discovery_owner"], "session-to-preserve")
                self.assertEqual(state["discovery_selected_job"], "finished-ode-run")
                self.assertEqual(state["epde_appearance"], "Dark")
                self.assertEqual(state["catalog_page"], 2)

    def test_reselecting_same_record_still_requires_new_approval(self):
        state = {"lab_stage": 3, "lab_confirm_science": True,
                 "lab_draft_input": {"record_name": "ns"}}
        launch_from_catalog(state, "ns")
        self.assertEqual(state.get("lab_stage"), 1)
        self.assertFalse(state.get("lab_confirm_science", False))
        self.assertNotIn("lab_draft_input", state)

    def test_invalid_selection_does_not_destroy_current_experiment(self):
        from support.discovery_lab.wizard_state import begin_catalog_experiment
        for invalid in (None, "", "  ", 23):
            with self.subTest(selection=invalid):
                state = {"lab_stage": 3, "lab_confirm_science": True}
                original = dict(state)
                with self.assertRaises(ValueError):
                    begin_catalog_experiment(state, invalid)
                self.assertEqual(state, original)

    def test_archive_defaults_to_deployed_repository_and_commit(self):
        from support.discovery_lab.portable import repository_source
        revision = "a" * 40
        with tempfile.TemporaryDirectory() as folder:
            with patch("epde_bench.paths.REPO_ROOT", Path(folder)):
                with patch.dict(os.environ, {"EPDE_REPOSITORY_REVISION": revision}, clear=True):
                    result = repository_source()
        self.assertEqual(result["url"], "https://github.com/MakSoS1/EPDE")
        self.assertEqual(result["revision"], revision)
        self.assertEqual(result["archive_url"],
                         f"https://github.com/MakSoS1/EPDE/archive/{revision}.zip")

    def test_archive_explicit_fork_override_is_preserved(self):
        from support.discovery_lab.portable import repository_source
        revision = "b" * 40
        with tempfile.TemporaryDirectory() as folder:
            with patch("epde_bench.paths.REPO_ROOT", Path(folder)):
                with patch.dict(os.environ, {"EPDE_REPOSITORY_REVISION": revision,
                                             "EPDE_REPOSITORY_URL": "https://github.com/emil817/EPDE_demo/"}, clear=True):
                    result = repository_source()
        self.assertEqual(result["archive_url"],
                         f"https://github.com/emil817/EPDE_demo/archive/{revision}.zip")


if __name__ == "__main__":
    unittest.main()
