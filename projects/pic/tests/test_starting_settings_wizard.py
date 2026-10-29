"""User-visible suggestion edits survive wizard navigation and reach review."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app"


def number(app, label):
    return next(w for w in app.number_input if w.label == label)


class StartingSettingsWizardTests(unittest.TestCase):
    def test_builtin_epde_only_does_not_render_the_upload_data_branch(self):
        app = AppTest.from_file(str(APP / "Home.py"), default_timeout=90).run()
        app.switch_page("views/10_Discovery_lab.py").run()
        app.session_state["lab_discovery_methods"] = ["epde"]
        app.run()
        self.assertFalse(app.exception)
        self.assertFalse(app.get("file_uploader"))

    def launch(self, source="Built-in equation"):
        app = AppTest.from_file(str(APP / "Home.py"), default_timeout=90).run()
        app.switch_page("views/10_Discovery_lab.py").run()
        app.session_state["lab_discovery_methods"] = ["epde"]
        app.session_state["lab_input_source"] = source
        if source == "Built-in equation":
            app.session_state["catalog_requested_record"] = "ode"
        app.run()
        self.assertFalse(app.exception)
        app.button(key="lab_next_data").click().run()
        self.assertFalse(app.exception)
        return app

    def test_manual_settings_survive_rerun_and_review_then_explicit_suggestion_resets_approval(self):
        app = self.launch()
        self.assertTrue(any(b.label == "Suggest settings" for b in app.button))
        original = number(app, "Population / candidates").value
        number(app, "Population / candidates").set_value(20).run()
        app.run()
        self.assertEqual(number(app, "Population / candidates").value, 20)
        app.button(key="lab_next_settings").click().run()
        self.assertEqual(app.session_state["lab_draft_config"]["overrides"]["search"]
                         ["evolution"]["population_size"], 20)
        app.checkbox(key="lab_confirm_science").check().run()
        app.button(key="lab_back_settings").click().run()
        self.assertEqual(number(app, "Population / candidates").value, 20)
        app.button(key="lab_advice_apply").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(number(app, "Population / candidates").value, original)
        self.assertFalse(app.session_state["lab_confirm_science"])
        self.assertFalse(app.session_state["lab_widgets_saved"]["lab_confirm_science"])

    def test_new_generated_numerical_input_does_not_inherit_old_manual_budget(self):
        app = self.launch("Generate an ODE")
        self.assertEqual(number(app, "Population / candidates").value, 12)
        number(app, "Population / candidates").set_value(20).run()
        first_context = app.session_state["lab_advice_context"]
        app.button(key="lab_back_data").click().run()
        next(w for w in app.text_input if w.label == "du/dt =").set_value("-u").run()
        app.button(key="lab_next_data").click().run()
        self.assertFalse(app.exception)
        self.assertNotEqual(app.session_state["lab_advice_context"], first_context)
        self.assertEqual(number(app, "Population / candidates").value, 12)
        self.assertFalse(app.session_state["lab_confirm_science"])

    def test_adding_a_method_keeps_the_existing_methods_manual_settings(self):
        app = self.launch()
        number(app, "Population / candidates").set_value(20).run()
        original_context = app.session_state["lab_advice_context"]
        app.button(key="lab_back_data").click().run()
        app.session_state["lab_discovery_methods"] = ["epde", "pdecomp"]
        app.run()
        app.button(key="lab_next_data").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["lab_advice_context"], original_context)
        self.assertEqual(number(app, "Population / candidates").value, 20)
        self.assertTrue(any("STLSQ" in w.label for w in app.number_input))

    def test_generated_equation_and_geometry_survive_back_to_data(self):
        app = self.launch("Generate an ODE")
        app.button(key="lab_back_data").click().run()
        app.text_input(key="lab_ode_rhs").set_value("-u").run()
        number(app, "End time").set_value(15.0).run()
        number(app, "Samples").set_value(201).run()
        app.button(key="lab_next_data").click().run()
        number(app, "Population / candidates").set_value(20).run()
        original_context = app.session_state["lab_advice_context"]
        app.button(key="lab_back_data").click().run()
        self.assertEqual(app.text_input(key="lab_ode_rhs").value, "-u")
        self.assertEqual(number(app, "End time").value, 15.0)
        self.assertEqual(number(app, "Samples").value, 201)
        app.button(key="lab_next_data").click().run()
        self.assertEqual(app.session_state["lab_advice_context"], original_context)
        self.assertEqual(number(app, "Population / candidates").value, 20)

    def test_submission_saves_advice_separately_from_the_final_edited_configuration(self):
        from support.discovery_lab import job_queue as queue
        # Keep the real queue serialization; suppress only the external manager
        # process because this test checks the UI-to-request boundary.
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(queue, "ROOT", Path(directory)), \
                patch.object(queue, "MANAGER_LOCK", Path(directory) / "manager.lock"), \
                patch.object(queue, "ensure_manager", return_value=None):
            app = self.launch()
            number(app, "Population / candidates").set_value(20).run()
            app.button(key="lab_next_settings").click().run()
            app.checkbox(key="lab_confirm_science").check().run()
            app.button(key="lab_start").click().run()
            self.assertFalse(app.exception)
            requests = list(Path(directory).glob("owners/*/jobs/*/request.json"))
            self.assertEqual(len(requests), 1)
            request = json.loads(requests[0].read_text())
        self.assertEqual(request["overrides"]["search"]["evolution"]["population_size"], 20)
        self.assertEqual(len(request["starting_suggestions"]), 1)
        advice = request["starting_suggestions"][0]
        self.assertEqual(advice["method"], "epde")
        self.assertEqual(advice["settings"]["search"]["evolution"]["population_size"], 16)


if __name__ == "__main__":
    unittest.main()
