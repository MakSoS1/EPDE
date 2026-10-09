"""The short evolution lesson executes the actual optimizer and records its front."""
import importlib
import math
import sys
import unittest
from pathlib import Path

PIC = Path(__file__).resolve().parents[1]
APP = PIC / 'app'
sys.path[:0] = [str(APP), str(PIC), str(PIC.parent.parent)]
from epde_bench import env
env.pin_blas_threads(1)


class OptimizerDemoTests(unittest.TestCase):
    def test_live_run_observes_epochs_and_recovers_the_known_oscillator(self):
        try:
            module = importlib.import_module('epde_bench.optimizer_demo')
        except ModuleNotFoundError:
            module = None
        self.assertIsNotNone(module, 'the live optimizer lesson needs a reusable runner')
        result = module.run_optimizer_demo()
        self.assertEqual([entry['epoch'] for entry in result['history']], [1, 2, 3])
        self.assertEqual([row['Epoch'] for row in result['rows']], [1, 2, 3])
        self.assertGreaterEqual(result['seconds'], 0.)
        for entry in result['history']:
            self.assertTrue(entry['front'])
            for solution in entry['front']:
                self.assertTrue(all(math.isfinite(v) for v in solution['obj_fun']))
                self.assertIn('=', solution['text_form'])
        from epde_bench.metrics import coefficient_error_best
        truth = ['-1.0625 * u{power: 1.0} + -0.5 * du/dx0{power: 1.0} '
                 '= d^2u/dx0^2{power: 1.0}']
        self.assertTrue(result['equations'])
        self.assertLess(coefficient_error_best(result['equations'], [truth]), 1e-10)


try:
    from streamlit.testing.v1 import AppTest
except ImportError:
    AppTest = None


@unittest.skipIf(AppTest is None, 'streamlit is not installed')
class OptimizerLessonTests(unittest.TestCase):
    def test_button_runs_and_displays_observed_history_and_equation(self):
        app = AppTest.from_file(str(APP / 'views/9_How_EPDE_works.py'), default_timeout=60)
        app.run()
        self.assertFalse(app.exception, [e.value for e in app.exception])
        button = next((b for b in app.button if b.label == 'Run short evolution'), None)
        self.assertIsNotNone(button, 'the lesson must offer a real short run')
        button.click().run()
        self.assertFalse(app.exception, [e.value for e in app.exception])
        self.assertTrue(app.dataframe)
        self.assertEqual(list(app.dataframe[0].value['Epoch']), [1, 2, 3])
        self.assertTrue(app.latex)
        self.assertIn('u_{tt}', app.latex[-1].value)
        self.assertIn('0.5', app.latex[-1].value)
        self.assertEqual(len(app.radio), 1, 'the existing stage navigation stays available')


if __name__ == '__main__':
    unittest.main()
