"""Isolated regression checks for the new demo-track interface.

Tests intentionally avoid EPDE/Streamlit runtime and any network access.
Run: python -m unittest discover -s projects/pic/tests -p 'test_discovery_lab.py' -v
"""
from __future__ import annotations
import importlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

APP = Path(__file__).resolve().parents[1] / 'app'
sys.path.insert(0, str(APP))

from support.discovery_lab.contract import normalize, llm_payload_valid
from support.discovery_lab.ode import compile_rhs, generate


class ContractTests(unittest.TestCase):
    GOOD = {'status':'ok', 'front':[['-0.5 * u{power: 1.0} = du/dx0{power: 1.0}']],
            'objectives':[[0.12, 2.0]], 'selected':['-0.5 * u{power: 1.0} = du/dx0{power: 1.0}'],
            'metrics':{'selected_index':0}}

    def test_native_epde_preserves_fields(self):
        rec = normalize(dict(self.GOOD, search_config={'objectives':{'second_objective':'instability'}}), 'epde')
        self.assertEqual(rec['method'], 'epde')
        self.assertEqual(rec['criterion_labels'][1], 'instability')
        self.assertFalse(rec['objectives_comparable_between_methods'])
        self.assertEqual(rec['front'], self.GOOD['front'])

    def test_single_pysindy_candidate_is_not_invented_pareto_front(self):
        rec = normalize(dict(self.GOOD, objectives=[None]), 'pdecomp')
        self.assertEqual(len(rec['front']), 1)
        self.assertIn('not reported', rec['criterion_labels'][1])

    def test_earlier_pysindy_baseline_is_not_an_app_method(self):
        with self.assertRaises(ValueError):
            normalize(self.GOOD, 'pysindy')

    def test_wrong_index_and_objective_are_rejected(self):
        for mutation in ({'metrics':{'selected_index':2}}, {'objectives':[[float('nan'), 2]]},
                         {'selected':['another = equation']}, {'front':[['arbitrary script']]}):
            with self.subTest(mutation=str(mutation)):
                with self.assertRaises(ValueError):
                    normalize({**self.GOOD, **mutation}, 'epde')

    def test_llm_must_name_its_own_measured_objectives(self):
        with self.assertRaises(ValueError):
            llm_payload_valid(dict(self.GOOD))
        rec = llm_payload_valid(dict(self.GOOD, criterion_labels=['heldout error', 'stability']))
        self.assertEqual(rec['method'], 'llm')

    def test_unsupported_llm_is_explicit(self):
        self.assertEqual(normalize({'status':'unsupported','reason':'not configured'}, 'llm')['front'], [])


class EquationGeneratorTests(unittest.TestCase):
    def test_safe_ode_generation(self):
        fn = compile_rhs('-0.5*u + sin(t)')
        self.assertAlmostEqual(fn(0, 2), -1.0)
        arrays = generate('-0.5*u + sin(t)', 3., 1., 101)
        self.assertEqual(arrays['var_u'].shape, (101,))
        self.assertAlmostEqual(arrays['var_u'][0], 1.0)

    def test_malicious_or_unbounded_syntax_is_rejected(self):
        bad = ['__import__("os").system("echo pwn")', 'open("/etc/passwd")',
               '(lambda: 1)()', '[t for t in range(10)]', 'u**u', 'u**99',
               '1e309', 't.__class__', 'sum([1,2])', 'u[0]', 't if t else u']
        for expr in bad:
            with self.subTest(expr=expr):
                with self.assertRaises(ValueError):
                    compile_rhs(expr)

    def test_unbounded_input_is_rejected(self):
        with self.assertRaises(ValueError):
            generate('u', 1, 1, 500000)


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        try:
            from support.discovery_lab import job_queue as queue
        except ModuleNotFoundError as exc:
            # The standalone overlay verification runner has no EPDE checkout;
            # real repository tests use the genuine epde_bench.paths module.
            if exc.name not in ('epde_bench', 'epde_bench.paths'):
                raise
            paths = types.ModuleType('epde_bench.paths')
            paths.RESULTS_DIR = Path(self.tmp.name)
            sys.modules['epde_bench'] = types.ModuleType('epde_bench')
            sys.modules['epde_bench.paths'] = paths
            from support.discovery_lab import job_queue as queue
            self.injected_paths = True
        else:
            self.injected_paths = False
        self.queue = queue
        self.old_root, self.old_lock = queue.ROOT, queue.MANAGER_LOCK
        queue.ROOT = Path(self.tmp.name)
        queue.MANAGER_LOCK = queue.ROOT / 'manager.lock'

    def tearDown(self):
        self.queue.ROOT, self.queue.MANAGER_LOCK = self.old_root, self.old_lock
        self.tmp.cleanup()
        if self.injected_paths:
            sys.modules.pop('epde_bench.paths', None)
            sys.modules.pop('epde_bench', None)

    def test_owners_are_isolated_and_job_ids_unforgeable(self):
        a, b = self.queue.new_owner(), self.queue.new_owner()
        self.assertNotEqual(a, b)
        request = {'method':'epde', 'source':'record', 'dataset':'ode', 'variant':'default'}
        with patch.object(self.queue, 'ensure_manager', return_value=None):
            jid = self.queue.submit(a, request)
        self.assertEqual(len(self.queue.list_jobs(a)), 1)
        self.assertEqual(self.queue.list_jobs(b), [])
        self.assertIsNone(self.queue.get_job(b, jid))
        with self.assertRaises(ValueError):
            self.queue.job_path('../etc', jid)
        with self.assertRaises(ValueError):
            self.queue.job_path(a, '../../secret')

    def test_pending_limit_and_cancellation(self):
        a = self.queue.new_owner()
        req = {'method':'pdecomp', 'source':'record', 'dataset':'ode', 'variant':'pysindy_pdecomp_fixed'}
        with patch.object(self.queue, 'ensure_manager', return_value=None):
            ids = [self.queue.submit(a, req) for _ in range(self.queue.MAX_OWNER_PENDING)]
            with self.assertRaisesRegex(ValueError, 'waiting or running'):
                self.queue.submit(a, req)
        self.assertTrue(self.queue.cancel(a, ids[1]))
        self.assertTrue((self.queue.job_path(a, ids[1]) / 'cancel').exists())

    def test_interrupted_jobs_delete_and_batch_limit(self):
        a = self.queue.new_owner()
        req = {'method': 'epde', 'source': 'record', 'dataset': 'ode', 'variant': 'default'}
        with patch.object(self.queue, 'ensure_manager', return_value=None):
            jid = self.queue.submit(a, req)
            path = self.queue.job_path(a, jid) / 'job.json'
            job = self.queue._read(path)
            job.update(state='running', pid=999999)       # neither manager nor worker alive
            self.queue._write(path, job)
            self.assertEqual(self.queue.list_jobs(a)[0]['state'], 'interrupted')
            self.assertTrue(self.queue.delete_job(a, jid))
            self.assertEqual(self.queue.list_jobs(a), [])
            for _ in range(self.queue.MAX_OWNER_PENDING - 1):
                self.queue.submit(a, req)
            with self.assertRaises(ValueError):            # a pair would exceed the limit
                self.queue.check_capacity(a, 'epde', count=2)


class ReferenceLawTests(unittest.TestCase):
    def test_generated_laws_in_epde_form(self):
        from support.discovery_lab.ode import reference_law, uses_time
        law, freq, _ = reference_law('-0.5*u + sin(t)')
        self.assertEqual(law, '-0.5 * u{power: 1.0} + 1.0 * sin{power: 1.0, freq: 1.0, dim: 0.0}'
                              ' = du/dx0{power: 1.0}')
        self.assertEqual(freq, 1.0)
        self.assertTrue(uses_time('-0.5*u + sin(t)'))
        self.assertFalse(uses_time('u - u**3'))
        self.assertIn('-1.0 * u{power: 3.0}', reference_law('u - u**3')[0])
        for unsupported in ('tanh(u)', 'sin(t) + cos(3*t)', 'u**4'):
            with self.subTest(expr=unsupported):
                self.assertIsNone(reference_law(unsupported)[0])


try:
    import epde_bench  # noqa: F401
    import pysindy  # noqa: F401
    FULL = True
except ImportError:                                       # standalone overlay checks
    FULL = False


@unittest.skipUnless(FULL, 'needs the EPDE checkout and PySINDy')
class WorkerAndPage(unittest.TestCase):
    """Real searches (PySINDy, seconds) through the worker, then the page on them."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        from support.discovery_lab import job_queue, worker
        cls.queue, cls.worker = job_queue, worker
        cls.old = job_queue.ROOT, job_queue.MANAGER_LOCK
        job_queue.ROOT = Path(cls.tmp.name)
        job_queue.MANAGER_LOCK = job_queue.ROOT / 'manager.lock'
        cls.owner = job_queue.new_owner()
        cls.ids = {}
        with patch.object(job_queue, 'ensure_manager', return_value=None):
            for name in ('ode', 'nir1_h11_biharmonic'):
                cls.ids[name] = job_queue.submit(cls.owner, {
                    'method': 'pdecomp', 'source': 'record', 'dataset': name,
                    'variant': 'pysindy_pdecomp_fixed' if name == 'ode' else 'pysindy_pdecomp_native',
                    'noise': 0.0, 'seed': 0, 'display': {'label': name, 'input_key': name}})
            from support.discovery_lab.ode import generate, reference_law
            from epde_bench.config import read_yaml
            from epde_bench.paths import CONFIG_DIR
            search = dict(read_yaml(CONFIG_DIR / '_base.yaml')['search'])
            search['domain'] = {'boundary_width': 'auto'}
            search['preprocessing'] = dict(search['preprocessing'], max_deriv_order=[1])
            search['search_space'] = dict(search['search_space'], tokens=[])
            settings = {'seed': 0, 'noise': 0.0, 'search': search,
                        'reference_truth': [reference_law('u - u**3')[0]], 'label': 'generated'}
            data, config = job_queue.save_upload(cls.owner, generate('u - u**3', 10, 0.1, 301), settings)
            cls.ids['generated'] = job_queue.submit(cls.owner, {
                'method': 'pdecomp', 'source': 'upload', 'dataset': 'custom', 'data_path': str(data),
                'config_path': str(config), 'display': {'label': 'generated', 'input_key': 'gen'}})
        for jid in cls.ids.values():
            folder = job_queue.job_path(cls.owner, jid)
            cls.worker.main(folder)
            job = job_queue.get_job(cls.owner, jid)
            job.update(state='finished', started=job['created'], ended=job['created'] + 1)
            job_queue._write(folder / 'job.json', job)

    @classmethod
    def tearDownClass(cls):
        cls.queue.ROOT, cls.queue.MANAGER_LOCK = cls.old
        cls.tmp.cleanup()

    def result(self, name):
        return self.queue.result(self.owner, self.ids[name])

    def test_ode_record_is_found_and_solved(self):
        record = self.result('ode')
        self.assertEqual(record['status'], 'ok')
        self.assertTrue(record['metrics']['success_selected'])
        self.assertGreater(record['evaluation']['solution']['r2']['u'], 0.99)

    def test_fourth_order_pde_is_solved(self):
        record = self.result('nir1_h11_biharmonic')
        solution = record['evaluation']['solution']
        self.assertIsNotNone(solution, record['evaluation'])
        self.assertGreater(solution['r2']['u'], 0.99)
        self.assertEqual(len(solution['simulated']['u']), len(solution['t']))

    def test_generated_ode_is_scored_against_its_law(self):
        record = self.result('generated')
        self.assertEqual(record['problem']['truth'][0].split('=')[1].strip(), 'du/dx0{power: 1.0}')
        self.assertTrue(record['metrics']['success_selected'])

    def test_page_renders_results_and_comparison(self):
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError:
            self.skipTest('streamlit is not installed')
        with patch.dict(os.environ, {}):
            app = AppTest.from_file(str(APP / 'views' / '10_Discovery_lab.py'), default_timeout=120)
            app.query_params['session'] = self.owner
            app.run()
        self.assertFalse(app.exception, [e.value for e in app.exception])
        self.assertEqual([t.label for t in app.tabs], ['① New experiment', '② Results', '③ Compare'])
        self.assertTrue(app.success)                      # known law found
        self.assertTrue(app.metric)


if __name__ == '__main__':
    unittest.main()
