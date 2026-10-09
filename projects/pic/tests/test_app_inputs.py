import sys
import unittest
from pathlib import Path
import tempfile
import numpy as np
import yaml
PIC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PIC / 'app'), str(PIC), str(PIC.parent.parent)]
from support.custom_run import problem_from_npz
from epde_bench.config import deep_merge

class Inputs(unittest.TestCase):
    def test_switch_discards_previous_builder_arguments(self):
        base = {'preprocessing': {'default_preprocessor_type': 'poly', 'preprocessor_kwargs': {'poly_order': 3}}}
        for method in ('FD', 'spectral', 'ANN'):
            result = deep_merge(base, {'preprocessing': {'default_preprocessor_type': method}})
            self.assertEqual(result['preprocessing']['preprocessor_kwargs'], {})

    def test_invalid_uploads_rejected_at_boundary(self):
        good = {'axis_0': np.arange(4.), 'var_u': np.arange(4.)}
        bad = [dict(good, axis_0=[0, 0, 1, 2]), dict(good, axis_0=[0, 1, 2, np.nan]),
               dict(good, axis_0=[3, 2, 1, 0]), dict(good, var_u=np.arange(3.)),
               {'axis_0': np.arange(4.), 'var_bad name': np.arange(4.)},
               {'axis_0': [0.], 'var_u': [1.]}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'data.npz'
            for arrays in bad:
                np.savez(path, **arrays)
                with self.subTest(arrays=list(arrays)), self.assertRaises(ValueError):
                    problem_from_npz(path)

    def test_multiple_variables_explicit_coordinates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'data.npz'
            np.savez(path, axis_0=[0., .1, .4], axis_1=[2., 4.], axis_names=['t', 'x'],
                     var_u=np.ones((3, 2)), var_v=np.zeros((3, 2)))
            problem = problem_from_npz(path)
            self.assertEqual(problem.variables, ['u', 'v'])
            self.assertEqual(problem.shape, (3, 2))

    def test_search_page_handles_missing_darcy_data(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(PIC / 'app/views/5_Run_search.py'), default_timeout=120).run()
        app.selectbox[0].set_value('darcy').run()
        self.assertFalse(app.exception, [e.value for e in app.exception])
        self.assertTrue(app.error)
        start = next(b for b in app.button if b.label == 'Start')
        self.assertTrue(start.disabled, 'a record without data cannot be started')

    def test_search_options_turn_into_overrides(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(PIC / 'app/views/5_Run_search.py'), default_timeout=120).run()
        self.assertFalse(app.exception, [e.value for e in app.exception])
        self.assertIn('# none', [c.value for c in app.code])
        effort = next(r for r in app.radio if r.label == 'search effort')
        effort.set_value('Quick').run()
        derivatives = next(r for r in app.radio if r.label == 'derivatives')
        derivatives.set_value('poly').run()
        changes = yaml.safe_load(next(c.value for c in app.code if 'search:' in c.value))
        self.assertEqual(changes['search']['preprocessing']['default_preprocessor_type'], 'poly')
        self.assertEqual(changes['search']['evolution'], {'population_size': 8, 'training_epochs': 2})

    def test_pendulum_derivatives_compute_after_poly_to_fd(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(PIC / 'app/views/3_Derivatives.py'), default_timeout=120).run()
        app.selectbox[0].set_value('pend_single').run()
        app.multiselect[0].set_value(['FD']).run()
        app.button[0].click().run()
        self.assertFalse(app.exception, [e.value for e in app.exception])
        self.assertTrue(app.metric)

    def test_custom_run_records_content_and_environment_without_mutating_upload(self):
        import hashlib
        import json
        import yaml
        from unittest.mock import patch
        from support import custom_run
        from support.viewer import record_problem
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, cfg, out = root / 'input.npz', root / 'settings.yaml', root / 'result.json'
            np.savez(data, axis_0=np.linspace(0, 1, 21), var_u=np.linspace(1, 2, 21))
            settings = {'seed': 2, 'noise': 0, 'search': {'preprocessing': {'max_deriv_order': [1]}}}
            cfg.write_text(yaml.safe_dump(settings))
            original = data.read_bytes()
            with patch('epde_bench.runner.discover', return_value=(None, [['1.0 = du/dx0{power: 1.0}']], [[0., 1.]], .01)):
                self.assertEqual(custom_run.main(['--data', str(data), '--config', str(cfg), '--out', str(out)]), 0)
                record = json.loads(out.read_text())
                self.assertEqual(record.get('data_sha256'), hashlib.sha256(original).hexdigest())
                self.assertIn('python', record['environment'])
                self.assertIn('source_sha256', record['identity'])
                self.assertEqual(data.read_bytes(), original)
                record_problem(record)
                identity = record['identity']
                cfg.write_text(yaml.safe_dump(dict(reversed(list(settings.items()))), sort_keys=False))
                custom_run.main(['--data', str(data), '--config', str(cfg), '--out', str(out)])
                self.assertEqual(identity, json.loads(out.read_text())['identity'])
            np.savez(data, axis_0=np.linspace(0, 1, 21), var_u=np.linspace(2, 3, 21))
            with self.assertRaisesRegex(ValueError, 'changed'):
                record_problem(record)
