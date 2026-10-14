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
        rec = normalize(self.GOOD, 'pysindy')
        self.assertEqual(len(rec['front']), 1)
        self.assertIn('active terms', rec['criterion_labels'][1])

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
            from support.discovery_lab import queue
        except ModuleNotFoundError as exc:
            # The standalone overlay verification runner has no EPDE checkout;
            # real repository tests use the genuine epde_bench.paths module.
            if exc.name not in ('epde_bench', 'epde_bench.paths'):
                raise
            paths = types.ModuleType('epde_bench.paths')
            paths.RESULTS_DIR = Path(self.tmp.name)
            sys.modules['epde_bench'] = types.ModuleType('epde_bench')
            sys.modules['epde_bench.paths'] = paths
            from support.discovery_lab import queue
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
        req = {'method':'pysindy', 'source':'record', 'dataset':'ode', 'variant':'pysindy'}
        with patch.object(self.queue, 'ensure_manager', return_value=None):
            ids = [self.queue.submit(a, req) for _ in range(3)]
            with self.assertRaisesRegex(ValueError, '3 pending'):
                self.queue.submit(a, req)
        self.assertTrue(self.queue.cancel(a, ids[1]))
        self.assertTrue((self.queue.job_path(a, ids[1]) / 'cancel').exists())


if __name__ == '__main__':
    unittest.main()
