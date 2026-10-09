"""Execute the displayed public API examples and check their existing UI placement."""
import os
from pathlib import Path
import subprocess
import sys
import unittest

PIC = Path(__file__).resolve().parents[1]
APP = PIC / 'app'
sys.path[:0] = [str(APP), str(PIC), str(PIC.parent.parent)]
from support.interfaces import EXAMPLES


class InterfaceExamples(unittest.TestCase):
    def test_displayed_examples_execute_in_independent_processes(self):
        assertions = {
            'settings': 'assert search.config.evolution.training_epochs == 3',
            'domain': 'assert (domain_id, trajectory_id, u.shape) == (0, 0, (121,))',
            'preprocessing': 'assert field.shape == (121,) and derivatives.shape == (121, 2)',
            'build': "assert problem.variables == ['u'] and trajectory is not None",
            'pool': 'assert search.pool is not None',
            'fit': 'assert texts and len(texts) == len(objectives)',
            'optimizer': "assert [entry['epoch'] for entry in history] == [1, 2, 3]\n"
                         "assert all(entry['front'] for entry in history)",
        }
        for key, example in EXAMPLES.items():
            with self.subTest(interface=key):
                # A separate process also respects EPDE's global-cache lifecycle.
                prelude = ("import os\nos.environ['OPENBLAS_NUM_THREADS'] = '1'\n"
                           "os.environ['OMP_NUM_THREADS'] = '1'\n"
                           'from epde_bench import env\nenv.seed_everything(7)\n')
                environ = dict(os.environ)
                environ['PYTHONPATH'] = os.pathsep.join((str(PIC), str(PIC.parent.parent)))
                result = subprocess.run([sys.executable, '-c', prelude + example.code + assertions[key]],
                                        env=environ, cwd=PIC.parent.parent, capture_output=True,
                                        text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_existing_stage_selector_shows_matching_interfaces(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(APP / 'views/9_How_EPDE_works.py'), default_timeout=60).run()
        self.assertFalse(app.exception, [e.value for e in app.exception])
        self.assertEqual(len(app.radio), 1)
        self.assertEqual(app.code[0].value, EXAMPLES['settings'].code.rstrip())
        app.radio[0].set_value('evolution').run()
        self.assertFalse(app.exception, [e.value for e in app.exception])
        self.assertEqual(app.code[0].value, EXAMPLES['optimizer'].code.rstrip())
        self.assertTrue(any(b.label == 'Run short evolution' for b in app.button))

    def test_derivatives_page_shows_preprocessing_and_trajectory_examples(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(APP / 'views/3_Derivatives.py'), default_timeout=120).run()
        self.assertFalse(app.exception, [e.value for e in app.exception])
        displayed = [code.value for code in app.code]
        self.assertIn(EXAMPLES['preprocessing'].code.rstrip(), displayed)
        self.assertIn(EXAMPLES['domain'].code.rstrip(), displayed)


if __name__ == '__main__':
    unittest.main()
