"""Editor behavior: explicit budget changes, manual edits and method isolation.

The UI is real Streamlit. PDEComp's settings lookup is replaced only where it
would start an external scientific Python process; rendering and validation
remain the production implementations.
"""
from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

PIC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PIC), str(PIC / "app"), str(PIC.parents[1])]


EPDE_EDITOR = """
import numpy as np
import streamlit as st
from support.discovery_lab.lab_settings import render_settings
arrays = {'axis_0': np.linspace(0, 1, 101), 'var_u': np.exp(-np.linspace(0, 1, 101))}
result = render_settings(record_name=None, problem=None, arrays=arrays,
    space={'time_terms': False}, source='Generate an ODE', methods=['epde'],
    noise=0, seed=0, label='Decay', origin='generated_ode', reference=None)
st.session_state['editor_result'] = result
"""

REAPPLY = """
import streamlit as st
from support.discovery_lab.advice_state import request_suggestions
st.button('Apply suggestions', on_click=lambda: request_suggestions(st.session_state))
"""


def number(app, label):
    return next(item for item in app.number_input if item.label == label)


def group(app, label):
    return next(item for item in app.get("button_group") if item.label == label)


class ExplicitEditorChoices(unittest.TestCase):
    def assert_valid(self, app):
        self.assertFalse(app.exception, [item.message for item in app.exception])

    def test_custom_budget_keeps_the_values_the_user_just_edited(self):
        app = AppTest.from_string(EPDE_EDITOR, default_timeout=40).run()
        self.assert_valid(app)
        number(app, "Population / candidates").set_value(20).run()
        number(app, "Evolution epochs").set_value(7).run()
        self.assertEqual(group(app, "Search budget").value, "Custom")
        group(app, "Search budget").set_value("Custom").run()
        self.assert_valid(app)
        self.assertEqual(number(app, "Population / candidates").value, 20)
        self.assertEqual(number(app, "Evolution epochs").value, 7)
        search = app.session_state["editor_result"][2]["search"]
        self.assertEqual(search["evolution"], {"population_size": 20, "training_epochs": 7})

    def test_default_budget_restores_the_original_values_in_controls_and_saved_state(self):
        app = AppTest.from_string(EPDE_EDITOR, default_timeout=40).run()
        number(app, "Population / candidates").set_value(20).run()
        number(app, "Evolution epochs").set_value(7).run()
        self.assertIn("Default", group(app, "Search budget").options)
        group(app, "Search budget").set_value("Quick").run()
        group(app, "Search budget").set_value("Default").run()
        self.assert_valid(app)
        self.assertEqual(number(app, "Population / candidates").value, 12)
        self.assertEqual(number(app, "Evolution epochs").value, 4)
        saved = app.session_state["lab_widgets_saved"]
        self.assertEqual(saved[number(app, "Population / candidates").key], 12)
        self.assertEqual(saved[number(app, "Evolution epochs").key], 4)

    def test_fixed_pde_controls_keep_geometry_after_optimizer_selection(self):
        from support.discovery_lab import pysindy_method
        base = {
            "optimizer": {"type": "STLSQ", "threshold": 0.1, "alpha": 0.05},
            "crop": 1,
            "library": {"data_fun_pow": 3, "equation_factors_max_number": 2,
                        "include_bias": True, "max_deriv_order": [2, 4],
                        "custom_tokens": ["x"]},
            "targets": [{"axis": "t", "order": 1}],
        }
        source = """
import streamlit as st
from support.discovery_lab.pysindy_method import settings_editor
st.session_state['editor_result'] = settings_editor(None, 'fixed', 'pde-test', kind='pde_1d')
"""
        with patch.object(pysindy_method, "pdecomp_settings", return_value=base):
            app = AppTest.from_string(source, default_timeout=40).run()
            self.assert_valid(app)
            self.assertTrue(any(item.label == "Highest space derivative" for item in app.number_input))
            tokens = next(item for item in app.multiselect if item.label == "Extra library terms")
            self.assertIn("x", tokens.options)
            self.assertEqual(tokens.value, ["x"])
            group(app, "Sparse optimizer").set_value("FROLS").run()
            number(app, "Highest space derivative").set_value(3).run()
            self.assert_valid(app)
            changes = app.session_state["editor_result"]
            self.assertEqual(changes["optimizer"]["type"], "FROLS")
            self.assertEqual(changes["library"]["max_deriv_order"], [2, 3])

    def test_suggested_epde_derivatives_reach_config_and_preserve_later_edits(self):
        context = {"dataset": None, "kind": "ode", "shape": [101], "variable_count": 1,
                   "noise_percent": 2.0, "uniform_axes": [True], "heavy": False}
        source = REAPPLY + EPDE_EDITOR.replace("noise=0, seed=0", "noise=2, seed=0").replace(
            "reference=None)", f"reference=None, advice_context={context!r})")
        app = AppTest.from_string(source, default_timeout=40).run()
        self.assert_valid(app)
        self.assertEqual(group(app, "Search budget").value, "Suggested")
        search = app.session_state["editor_result"][2]["search"]
        self.assertEqual(search["preprocessing"]["default_preprocessor_type"], "poly")
        self.assertEqual(search["preprocessing"]["preprocessor_kwargs"]["polynomial_window"], 9)
        group(app, "Numerical derivatives").set_value("FD").run()
        number(app, "Population / candidates").set_value(20).run()
        app.run()
        self.assert_valid(app)
        search = app.session_state["editor_result"][2]["search"]
        self.assertEqual(search["preprocessing"]["default_preprocessor_type"], "FD")
        self.assertEqual(search["evolution"]["population_size"], 20)
        self.assertIn("epde:data:default", app.session_state["lab_advice_reports"])
        previous_keys = {item.key for item in app.number_input} | {
            group(app, "Search budget").key, group(app, "Numerical derivatives").key}
        next(item for item in app.button if item.label == "Apply suggestions").click().run()
        self.assert_valid(app)
        fresh_keys = {item.key for item in app.number_input} | {
            group(app, "Search budget").key, group(app, "Numerical derivatives").key}
        self.assertTrue(previous_keys.isdisjoint(fresh_keys),
                        "Reapply must mount new controls, not reuse stale browser widget identities")
        self.assertEqual(group(app, "Search budget").value, "Suggested")
        self.assertEqual(number(app, "Population / candidates").value, 12)
        self.assertEqual(group(app, "Numerical derivatives").value, "poly")

    def test_pysindy_suggested_trim_is_exported_and_explicit_default_reset_persists(self):
        from support.discovery_lab import pysindy_method
        base = {"optimizer": {"type": "STLSQ", "threshold": 0.1, "alpha": 0.05},
                "polynomial_degree": 3, "boundary_trim": 20,
                "differentiation": {"order": 4, "periodic": False}}
        context = {"dataset": None, "kind": "ode", "shape": [20], "variable_count": 1,
                   "noise_percent": 0.0, "uniform_axes": [True], "heavy": False}
        source = f"""
import streamlit as st
from support.discovery_lab.pysindy_method import settings_editor
from support.discovery_lab.advice_state import request_suggestions
st.button('Apply suggestions', on_click=lambda: request_suggestions(st.session_state))
st.session_state['editor_result'] = settings_editor(None, 'native', 'tiny', kind='ode',
                                                   advice_context={context!r})
"""
        with patch.object(pysindy_method, "pdecomp_settings", return_value=base):
            app = AppTest.from_string(source, default_timeout=40).run()
            self.assert_valid(app)
            self.assertEqual(number(app, "Boundary trim, points").value, 6)
            self.assertEqual(app.session_state["editor_result"]["boundary_trim"], 6)
            number(app, "Boundary trim, points").set_value(3).run()
            app.run()
            self.assertEqual(app.session_state["editor_result"]["boundary_trim"], 3)
            app.button(key="pdecomp_reset_native_tiny").click().run()
            app.run()
            self.assert_valid(app)
            self.assertEqual(number(app, "Boundary trim, points").value, 20)
            self.assertNotIn("boundary_trim", app.session_state["editor_result"])
            previous_keys = {item.key for item in app.number_input}
            next(item for item in app.button if item.label == "Apply suggestions").click().run()
            self.assertTrue(previous_keys.isdisjoint({item.key for item in app.number_input}))
            self.assertEqual(number(app, "Boundary trim, points").value, 6)
            self.assertEqual(app.session_state["editor_result"]["boundary_trim"], 6)

    def test_native_suggestions_and_manual_budgets_reach_the_method_configuration(self):
        from support.discovery_lab import pdecomp_methods
        context = {"dataset": None, "kind": "pde_1d", "shape": [21, 31], "variable_count": 1,
                   "noise_percent": 0.0, "uniform_axes": [True, True], "heavy": False}
        fixtures = (
            ("deepmod", {"max_iterations": 5000, "max_samples": 10000, "learning_rate": 0.002,
                         "threshold": 0.1, "poly_order": 3, "diff_order": 4},
             "Training iterations", "max_iterations", 250, 25, "Maximum training samples", 651),
            ("discover", {"n_samples": 5000, "batch_size": 250, "epsilon": 0.05,
                          "max_length": 15, "max_add_count": 8},
             "Expression samples", "n_samples", 1000, 120, "Controller batch size", 100),
        )
        for method, base, control, setting, suggested, edited, second, second_value in fixtures:
            with self.subTest(method=method), patch.object(pdecomp_methods, "starting_settings", return_value=base):
                source = REAPPLY + f"""
import numpy as np
import streamlit as st
from support.discovery_lab.pdecomp_methods import render_settings
arrays = {{'axis_0': np.linspace(0, 1, 21), 'axis_1': np.linspace(0, 1, 31),
          'var_u': np.ones((21, 31))}}
st.session_state['editor_result'] = render_settings({method!r}, arrays=arrays,
    source='Upload your data', advice_context={context!r})
"""
                app = AppTest.from_string(source, default_timeout=40).run()
                self.assert_valid(app)
                self.assertEqual(group(app, "Run budget").value, "Suggested")
                self.assertEqual(number(app, control).value, suggested)
                self.assertEqual(number(app, second).value, second_value)
                self.assertEqual(app.session_state["editor_result"]["overrides"][setting], suggested)
                number(app, control).set_value(edited).run()
                app.run()
                self.assertEqual(app.session_state["editor_result"]["overrides"][setting], edited)
                previous_keys = {item.key for item in app.number_input} | {group(app, "Run budget").key}
                selectors = {label: (group(app, label).key, group(app, label).value)
                             for label in ("Protocol", "Target time derivative")}
                next(item for item in app.button if item.label == "Apply suggestions").click().run()
                fresh_keys = {item.key for item in app.number_input} | {group(app, "Run budget").key}
                self.assertTrue(previous_keys.isdisjoint(fresh_keys))
                self.assertEqual(number(app, control).value, suggested)
                for label, original in selectors.items():
                    self.assertEqual((group(app, label).key, group(app, label).value), original)
                group(app, "Run budget").set_value("PDEComp profile").run()
                self.assertEqual(number(app, control).value, 5000)
                group(app, "Run budget").set_value("Suggested").run()
                self.assert_valid(app)
                self.assertEqual(number(app, control).value, suggested)
                self.assertEqual(number(app, second).value, second_value)
                self.assertEqual(app.session_state["lab_widgets_saved"][number(app, control).key], suggested)

    def test_heavy_suggestion_exports_edited_budget_without_loading_scientific_data(self):
        from support.discovery_lab.portable import make_manifest
        context = {"dataset": "ns", "kind": "pde_2d", "shape": [], "variable_count": None,
                   "noise_percent": 0.0, "uniform_axes": [], "heavy": True}
        source = REAPPLY + f"""
import streamlit as st
from support.discovery_lab.heavy_settings import render_heavy_epde
st.session_state['editor_result'] = render_heavy_epde('ns', advice_context={context!r})
"""
        with patch("epde_bench.datasets.load", side_effect=AssertionError("Heavy data must stay unloaded")):
            app = AppTest.from_string(source, default_timeout=40).run()
            self.assert_valid(app)
            number(app, "Population / candidates").set_value(8).run()
            app.run()
            self.assertEqual(number(app, "Population / candidates").value, 8)
            manifest = make_manifest("ns", 0.0, 0, [app.session_state["editor_result"]])
            previous_keys = {item.key for item in app.number_input}
            next(item for item in app.button if item.label == "Apply suggestions").click().run()
            self.assert_valid(app)
            self.assertTrue(previous_keys.isdisjoint({item.key for item in app.number_input}))
            self.assertEqual(number(app, "Population / candidates").value, 48)
        self.assertEqual(manifest["dataset"], "ns")
        self.assertEqual(manifest["runs"][0]["method"], "epde")
        self.assertEqual(manifest["runs"][0]["overrides"]["search"]["evolution"]["population_size"], 8)
        self.assertIn("epde:ns:default", app.session_state["lab_advice_reports"])


if __name__ == "__main__":
    unittest.main()
