"""Recommendations respect real scientific/resource gates and wizard ownership."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

import numpy as np

from epde_bench.config import load_config
from support.discovery_lab.execution_policy import classify_epde
from support.discovery_lab.lab_settings import _custom_base, validate_public_search
from support.discovery_lab.starting_settings import summarize_input, suggest_settings
from support.discovery_lab.advice_state import prepare_context, request_suggestions, save_report


class StartingSettingsTests(unittest.TestCase):
    def test_small_input_keeps_profile_and_larger_input_fits_hosted_work_gate(self):
        original = {"search": _custom_base(0, generated=True)}
        unchanged = deepcopy(original)
        for shape in ((301,), (100, 80), (11, 11)):
            with self.subTest(shape=shape):
                context = {"dataset": None, "kind": "ode" if len(shape) == 1 else "pde_1d",
                           "shape": list(shape), "variable_count": 1, "noise_percent": 0.0,
                           "uniform_axes": [True] * len(shape), "heavy": False}
                base = {"search": _custom_base(len(shape) - 1, generated=True)}
                report = suggest_settings("epde", base, context)
                search = report["settings"]["search"]
                validate_public_search(search, shape)
                allowed, reason, _ = classify_epde(None, shape, 1, search)
                self.assertTrue(allowed, reason)
                if shape == (301,):
                    self.assertEqual(search["evolution"]["population_size"], 12)
                    self.assertEqual(search["evolution"]["training_epochs"], 4)
                self.assertTrue(report["reasons"])
        self.assertEqual(original, unchanged)

    def test_declared_noise_changes_derivatives_without_changing_candidate_vocabulary(self):
        base = {"search": _custom_base(1)}
        context = {"shape": [51, 31], "variable_count": 1, "noise_percent": 3.0,
                   "uniform_axes": [True, True], "heavy": False}
        report = suggest_settings("epde", base, context)
        result = report["settings"]["search"]
        self.assertEqual(result["preprocessing"]["default_preprocessor_type"], "poly")
        self.assertEqual(result["preprocessing"]["max_deriv_order"], [2, 3])
        self.assertEqual(result["search_space"], base["search"]["search_space"])
        validate_public_search(result, (51, 31))
        context["noise_percent"] = 0.0
        clean = suggest_settings("epde", base, context)["settings"]["search"]
        self.assertEqual(clean["preprocessing"]["default_preprocessor_type"], "FD")

    def test_noisy_stored_auto_derivative_profile_is_resolved_before_validation(self):
        base = load_config("burgers", "default")
        self.assertEqual(base["search"]["preprocessing"]["max_deriv_order"], "auto")
        context = {"dataset": "burgers", "shape": [81, 61], "variable_count": 1,
                   "noise_percent": 2.0, "uniform_axes": [True, True], "heavy": False}
        report = suggest_settings("epde", base, context)
        proposed = report["settings"]["search"]
        self.assertEqual(proposed["preprocessing"]["max_deriv_order"], [2, 4])
        validate_public_search(proposed, (81, 61))
        allowed, reason, _ = classify_epde("burgers", (81, 61), 1, proposed)
        self.assertTrue(allowed, reason)

    def test_too_short_grid_does_not_receive_invalid_polynomial_fit(self):
        base = {"search": _custom_base(0)}
        base["search"]["preprocessing"]["max_deriv_order"] = [6]
        context = {"shape": [10], "variable_count": 1, "noise_percent": 5.0,
                   "uniform_axes": [True], "heavy": False}
        result = suggest_settings("epde", base, context)["settings"]["search"]
        self.assertEqual(result["preprocessing"]["default_preprocessor_type"], "FD")
        validate_public_search(result, (10,))

    def test_native_budgets_use_available_samples_and_keep_optimizer_semantics(self):
        context = {"dataset": None, "shape": [20, 20], "variable_count": 1, "noise_percent": 0,
                   "uniform_axes": [True, True], "heavy": False}
        base = {"max_iterations": 5000, "max_samples": 10000,
                "learning_rate": 0.002, "threshold": 0.1, "poly_order": 3, "diff_order": 2}
        report = suggest_settings("deepmod", base, context)
        self.assertEqual(report["settings"]["max_samples"], 400)
        self.assertEqual(report["settings"]["max_iterations"], 250)
        self.assertEqual(report["settings"]["threshold"], 0.1)
        fixed = {"direct_optimizer": {"type": "threshold", "threshold": 0.2}}
        self.assertEqual(suggest_settings("deepmod", fixed, context,
                                         protocol="fixed")["settings"], fixed)
        discover = {"n_samples": 5000, "batch_size": 250, "epsilon": 0.05,
                    "max_length": 15, "max_add_count": 8}
        proposed = suggest_settings("discover", discover, context)["settings"]
        self.assertLessEqual(proposed["batch_size"], proposed["n_samples"])
        self.assertEqual(proposed["epsilon"], 0.05)
        self.assertEqual(proposed["max_length"], 15)

    def test_sparse_regression_crop_fits_grid_and_threshold_is_not_guessed(self):
        context = {"shape": [12, 30], "variable_count": 1, "noise_percent": 2.0,
                   "uniform_axes": [True, True], "heavy": False}
        for protocol, field in (("native", "boundary_trim"), ("fixed", "crop")):
            with self.subTest(protocol=protocol):
                base = {field: 20, "optimizer": {"type": "STLSQ", "threshold": 0.007}}
                proposed = suggest_settings("pdecomp", base, context, protocol=protocol)
                self.assertLessEqual(proposed["settings"][field], 2)
                self.assertEqual(proposed["settings"]["optimizer"], base["optimizer"])

    def test_summary_excludes_reference_and_does_not_infer_noise_from_signal(self):
        arrays = {"axis_0": np.linspace(0, 1, 31), "var_u": np.sin(np.arange(31))}
        context = summarize_input(arrays=arrays, noise_percent=0.0)
        self.assertEqual(context["shape"], [31])
        self.assertEqual(context["kind"], "ode")
        self.assertEqual(context["uniform_axes"], [True])
        self.assertEqual(context["noise_percent"], 0.0)
        arrays["reference_truth"] = "private equation"
        arrays["var_u"] *= 100
        self.assertEqual(summarize_input(arrays=arrays), context)
        problem = SimpleNamespace(shape=(31,), kind="ode", variables=["u"],
                                  grids=[arrays["axis_0"]], truth="u_t=-u")
        first = summarize_input(problem=problem)
        problem.truth = "u_t=100*u"
        self.assertEqual(first, summarize_input(problem=problem))

    def test_heavy_input_is_metadata_only_and_keeps_full_budget(self):
        class NoNumericalAccess:
            def __getattribute__(self, name):
                raise AssertionError("Heavy numerical input must not be accessed")
        context = summarize_input(record_name="ns", problem=NoNumericalAccess(),
                                  arrays=NoNumericalAccess(), heavy=True)
        self.assertEqual(context["shape"], [])
        self.assertEqual(context["kind"], "pde_2d")
        base = load_config("ns", "default")
        proposed = suggest_settings("epde", base, context, hosted=False)
        self.assertEqual(proposed["settings"], base)
        self.assertTrue(any("metadata" in x.lower() for x in proposed["reasons"]))


class AdviceStateTests(unittest.TestCase):
    def state(self):
        return {"lab_advice_context": "old", "lab_stage": 2,
                "lab_input_source": "Generate an ODE", "lab_discovery_methods": ["epde"],
                "lab_rhs_u": "-u", "lab_draft_input": {"arrays": "preserved"},
                "heavy_noise_ns": 3.0, "heavy_seed_ns": 7,
                "lab_variant_Generate an ODE_None": "poly",
                "lab_pop_Generate an ODE_None_poly": 20,
                "lab_deepmod_data_protocol": "fixed", "lab_deepmod_data_target": 2,
                "lab_deepmod_data_fixed_threshold": 0.5,
                "portable_pdecomp_protocol_ns": "fixed",
                "lab_confirm_science": True, "lab_draft_config": {"stale": True},
                "lab_widgets_saved": {"lab_pop_Generate an ODE_None_poly": 20,
                                      "lab_confirm_science": True,
                                      "lab_input_source": "Generate an ODE"},
                "discovery_lab_owner": "owner", "discovery_selected_job": "finished"}

    def test_same_context_preserves_edits_and_changed_context_discards_configuration(self):
        state = self.state()
        self.assertFalse(prepare_context(state, "old"))
        self.assertEqual(state["lab_pop_Generate an ODE_None_poly"], 20)
        self.assertTrue(prepare_context(state, "new"))
        self.assertNotIn("lab_pop_Generate an ODE_None_poly", state)
        self.assertNotIn("lab_deepmod_data_fixed_threshold", state)
        self.assertNotIn("lab_draft_config", state)
        self.assertFalse(state["lab_confirm_science"])
        self.assertFalse(state["lab_widgets_saved"].get("lab_confirm_science"))
        self.assertEqual(state["lab_draft_input"], {"arrays": "preserved"})
        self.assertEqual(state["discovery_selected_job"], "finished")
        self.assertEqual(state["discovery_lab_owner"], "owner")
        self.assertEqual(state["lab_rhs_u"], "-u")
        self.assertEqual(state["heavy_noise_ns"], 3.0)
        self.assertEqual(state["heavy_seed_ns"], 7)

    def test_explicit_reapply_keeps_scientific_protocol_choices_but_invalidates_approval(self):
        state = self.state()
        save_report(state, "epde:data:poly", {"method": "epde", "settings": {}})
        request_suggestions(state)
        self.assertEqual(state["lab_variant_Generate an ODE_None"], "poly")
        self.assertEqual(state["lab_deepmod_data_protocol"], "fixed")
        self.assertEqual(state["lab_deepmod_data_target"], 2)
        self.assertEqual(state["portable_pdecomp_protocol_ns"], "fixed")
        self.assertNotIn("lab_deepmod_data_fixed_threshold", state)
        self.assertNotIn("lab_draft_config", state)
        self.assertFalse(state["lab_confirm_science"])
        self.assertFalse(state["lab_widgets_saved"].get("lab_confirm_science"))
        self.assertEqual(state["lab_advice_reports"], {})
        self.assertGreater(state["lab_advice_generation"], 0)


if __name__ == "__main__":
    unittest.main()
