"""Native method restrictions, configuration isolation, UI and real CPU smoke.

Set EPDE_TEST_NATIVE_METHODS=1 for the actual upstream NN/RNN invocations.
The short training budgets verify execution, not equation recovery accuracy.
"""
import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

PIC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PIC), str(PIC / 'app'), str(PIC.parents[1])]

from epde_bench import pdecomp_methods as methods
from epde_bench._pdecomp_methods_bridge import configure
from epde_bench.config import load_config
from epde_bench.problem import Problem
from support.discovery_lab import contract, portable


def heat():
    grids = tuple(np.meshgrid(np.linspace(0, 1, 21), np.linspace(0, 6, 31), indexing='ij'))
    return Problem('native_smoke', 'Heat', 'pde_1d', 'synthetic', grids,
                   {'u': np.exp(-grids[0]) * np.sin(grids[1])}, ('t', 'x'))


class NativeConfiguration(unittest.TestCase):
    def test_results_distinguish_training_and_reconstruction_derivatives(self):
        problem = heat()
        reply = {'result': {'targets': ['u_t'], 'features': [['u_xx']], 'coefficients': [[1.0]]},
                 'settings': {}, 'fit_seconds': 1}
        cases = (('deepmod', 'native', 'autograd'),
                 ('deepmod', 'fixed', 'finite_difference'),
                 ('discover', 'native', 'finite_difference'),
                 ('discover', 'fixed', 'finite_difference'))
        for method, protocol, differentiation in cases:
            with self.subTest(method=method, protocol=protocol), \
                    patch.object(methods, '_bridge', return_value=reply):
                _, _, _, info = methods.run(problem, problem.data, method, protocol=protocol)
                self.assertEqual(info['differentiation'], differentiation)
                self.assertEqual(info.get('reconstruction_differentiation'), 'finite_difference')

    def test_viewer_uses_recorded_reconstruction_derivatives_for_native_deepmod(self):
        from support.viewer import reconstruction_inputs
        record = {'search_config': {'preprocessing': {'max_deriv_order': [1, 2]}},
                  'baseline': {'differentiation': 'autograd',
                               'reconstruction_differentiation': 'finite_difference'}}
        _, search = reconstruction_inputs(record, heat())
        self.assertEqual(search['_reconstruction_differentiation'], 'finite_difference')
        self.assertNotIn('_reconstruction_differentiation', record['search_config'])

    def test_worker_reports_numerical_balance_separately_from_native_training(self):
        from epde_bench import reconstruction
        from support.discovery_lab.worker import evaluate
        problem = heat()
        record = {'status': 'ok', 'dataset': problem.name,
                  'selected': ['1.0 * d^2u/dx1^2{power: 1.0} = du/dx0{power: 1.0}'],
                  'search_config': {'preprocessing': {'max_deriv_order': [1, 2]},
                                    'domain': {'boundary_width': 1}},
                  'baseline': {'differentiation': 'autograd',
                               'reconstruction_differentiation': 'finite_difference'}}
        with patch('epde_bench.datasets.load', return_value=problem), \
                patch.object(reconstruction, 'simulate_pde', return_value=None):
            evaluation = evaluate(record)
        self.assertEqual(evaluation.get('differentiation'), 'finite_difference')
        self.assertGreater(evaluation['balance']['du/dx0{power: 1.0}'], 0.99)

    def test_portable_native_worker_has_no_hosted_time_limit(self):
        problem = heat()
        reply = {'result': {'targets': ['u_t'], 'features': [['u_xx']], 'coefficients': [[0.17]]},
                 'settings': {}, 'fit_seconds': 1}
        with patch.dict(os.environ, {'EPDE_LOCAL_FULL_RUN': '1'}), patch.object(methods, '_bridge', return_value=reply) as bridge:
            methods.run(problem, problem.data, 'deepmod')
        self.assertIsNone(bridge.call_args.kwargs['timeout'])

    def test_review_keeps_the_selected_seed(self):
        with patch.object(methods, '_bridge', return_value={'settings': {'seed': 7}}) as bridge:
            self.assertEqual(methods.describe('discover', 'ode', seed=7)['seed'], 7)
        self.assertEqual(bridge.call_args.args[0]['seed'], 7)

    def test_nn_workers_do_not_overlap_other_searches(self):
        from support.discovery_lab.manager import may_start
        self.assertTrue(may_start('deepmod', []))
        self.assertTrue(may_start('epde', ['pdecomp']))
        self.assertFalse(may_start('deepmod', ['epde']))
        self.assertFalse(may_start('discover', ['deepmod']))
        self.assertFalse(may_start('pdecomp', ['discover']))

    def test_no_silent_fixed_fallback(self):
        self.assertIsNotNone(methods.unsupported_reason('deepmod', 'ode', 1))
        self.assertIsNotNone(methods.unsupported_reason('deepmod', 'pde_1d', 1, orders={'u': 2}))
        self.assertIsNone(methods.unsupported_reason('deepmod', 'pde_1d', 1, orders={'u': 1}))
        self.assertIsNone(methods.unsupported_reason('deepmod', 'ode_system', 2, 'fixed'))
        self.assertIsNotNone(methods.unsupported_reason('discover', 'ode_system', 2))
        self.assertIsNotNone(methods.unsupported_reason('discover', 'pde_2d', 1))
        self.assertIsNone(methods.unsupported_reason('discover', 'ode', 1))

    def test_numerical_methods_reject_irregular_time_and_space_before_training(self):
        for method, protocol in (('discover', 'native'), ('discover', 'fixed'), ('deepmod', 'fixed')):
            for axis in (0, 1):
                with self.subTest(method=method, protocol=protocol, axis=axis):
                    coordinates = [np.linspace(0, 1, 21), np.linspace(0, 6, 31)]
                    coordinates[axis] = coordinates[axis] ** 2
                    grids = tuple(np.meshgrid(*coordinates, indexing='ij'))
                    problem = Problem('irregular', 'Irregular grid', 'pde_1d', 'synthetic', grids,
                                      {'u': np.exp(-grids[0]) * np.sin(grids[1])}, ('t', 'x'))
                    with patch.object(methods, '_bridge', side_effect=AssertionError('Training must not start')), \
                            self.assertRaisesRegex(methods.UnsupportedProblem, 'uniform.*' + ('t' if axis == 0 else 'x')):
                        methods.run(problem, problem.data, method, protocol=protocol)

    def test_numerical_methods_accept_uniform_float_grids(self):
        reply = {'result': {'targets': ['u_t'], 'features': [['u_xx']], 'coefficients': [[1.0]]},
                 'settings': {}, 'fit_seconds': 1}
        for dtype in (np.float32, np.float64):
            for method, protocol in (('discover', 'native'), ('discover', 'fixed'), ('deepmod', 'fixed')):
                with self.subTest(dtype=dtype, method=method, protocol=protocol):
                    coordinates = [np.linspace(0, 1, 1000, dtype=dtype), np.linspace(0, 6, 31, dtype=dtype)]
                    grids = tuple(np.meshgrid(*coordinates, indexing='ij'))
                    problem = Problem('uniform', 'Uniform grid', 'pde_1d', 'synthetic', grids,
                                      {'u': np.exp(-grids[0]) * np.sin(grids[1])}, ('t', 'x'))
                    with patch.object(methods, '_bridge', return_value=reply):
                        front, _, _, _ = methods.run(problem, problem.data, method, protocol=protocol)
                    self.assertIn('du/dx0', front[0][0])

    def test_uniform_float32_upload_stays_supported_after_coordinate_conversion(self):
        from support.custom_run import problem_from_npz
        t, x = np.linspace(0, 1, 1000, dtype=np.float32), np.linspace(0, 6, 31, dtype=np.float32)
        grids = np.meshgrid(t, x, indexing='ij')
        reply = {'result': {'targets': ['u_t'], 'features': [['u_xx']], 'coefficients': [[1.0]]},
                 'settings': {}, 'fit_seconds': 1}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'upload.npz'
            np.savez(path, axis_0=t, axis_1=x, axis_names=np.array(['t', 'x']),
                     var_u=np.exp(-grids[0]) * np.sin(grids[1]))
            problem = problem_from_npz(path)
        for method, protocol in (('discover', 'native'), ('discover', 'fixed'), ('deepmod', 'fixed')):
            with self.subTest(method=method, protocol=protocol), \
                    patch.object(methods, '_bridge', return_value=reply):
                try:
                    front, _, _, _ = methods.run(problem, problem.data, method, protocol=protocol)
                except methods.UnsupportedProblem as exc:
                    self.fail(f'Valid float32 upload was rejected after conversion: {exc}')
                self.assertIn('du/dx0', front[0][0])

    def test_float64_spacing_check_is_not_globally_relaxed_for_float32_rounding(self):
        t = np.linspace(0, 1, 1000)
        t[500] += 1e-7
        grids = tuple(np.meshgrid(t, np.linspace(0, 6, 31), indexing='ij'))
        problem = Problem('float64_grid', 'Irregular float64 grid', 'pde_1d', 'synthetic', grids,
                          {'u': np.exp(-grids[0]) * np.sin(grids[1])}, ('t', 'x'))
        for method, protocol in (('discover', 'native'), ('discover', 'fixed'), ('deepmod', 'fixed')):
            with self.subTest(method=method, protocol=protocol), \
                    patch.object(methods, '_bridge', side_effect=AssertionError('Training must not start')), \
                    self.assertRaisesRegex(methods.UnsupportedProblem, 'uniform t'):
                methods.run(problem, problem.data, method, protocol=protocol)

    def test_numerical_methods_reject_large_offset_float32_irregular_grid(self):
        steps = np.tile(np.array([0.125, 0.25], dtype=np.float32), 10)
        t = np.float32(1_000_000) + np.concatenate((np.zeros(1, dtype=np.float32), np.cumsum(steps)))
        grids = tuple(np.meshgrid(t, np.linspace(0, 6, 31, dtype=np.float32), indexing='ij'))
        problem = Problem('offset_grid', 'Irregular offset grid', 'pde_1d', 'synthetic', grids,
                          {'u': (grids[0] - t[0]) + grids[1] ** 2}, ('t', 'x'))
        for method, protocol in (('discover', 'native'), ('discover', 'fixed'), ('deepmod', 'fixed')):
            with self.subTest(method=method, protocol=protocol), \
                    patch.object(methods, '_bridge', side_effect=AssertionError('Training must not start')), \
                    self.assertRaisesRegex(methods.UnsupportedProblem, 'uniform t'):
                methods.run(problem, problem.data, method, protocol=protocol)

    def test_native_deepmod_preserves_coordinate_aware_irregular_input(self):
        grids = tuple(np.meshgrid(np.linspace(0, 1, 21) ** 2, np.linspace(0, 6, 31) ** 2, indexing='ij'))
        problem = Problem('irregular', 'Irregular grid', 'pde_1d', 'synthetic', grids,
                          {'u': np.exp(-grids[0]) * np.sin(grids[1])}, ('t', 'x'))
        reply = {'result': {'targets': ['u_t'], 'features': [['u_xx']], 'coefficients': [[1.0]]},
                 'settings': {}, 'fit_seconds': 1}
        with patch.object(methods, '_bridge', return_value=reply):
            _, _, _, info = methods.run(problem, problem.data, 'deepmod')
        self.assertEqual(info['differentiation'], 'autograd')

    def test_options_are_validated_before_execution(self):
        for method, changes in [('deepmod', {'max_iterations': True}),
                                ('deepmod', {'max_iterations': float('nan')}),
                                ('deepmod', {'max_samples': 100.5}),
                                ('deepmod', {'device': 'cuda'}),
                                ('discover', {'n_cores_batch': 100}),
                                ('discover', {'n_samples': -1}),
                                ('discover', {'function_set': ['exec']})]:
            with self.subTest(method=method, changes=changes), self.assertRaises(ValueError):
                methods.validate_overrides(method, changes)

    def test_native_settings_do_not_include_fixed_sparse_optimizer(self):
        config = SimpleNamespace(
            deepmod_params={'test': {'direct_optimizer': {'type': 'pdefind'}}},
            DEEPMOD_DEFAULTS={'direct_optimizer': {}},
            NATIVE_DEEPMOD_DEFAULTS={'max_iterations': 5000, 'threshold': 0.1},
            NATIVE_DEEPMOD_DATASET_OVERRIDES={'test': {'threshold': 0.2}},
            sindy_params={'test': {'targets': [{'name': 'u_t', 'order': 1}]}},
        )
        before = copy.deepcopy(config.deepmod_params)
        _, effective = configure(config, {'method': 'deepmod', 'filename': 'test',
                                          'protocol': 'native', 'seed': 7,
                                          'overrides': {'max_iterations': 2}})
        self.assertEqual(effective['seed'], 7)
        self.assertEqual(effective['threshold'], 0.2)
        self.assertEqual(effective['max_iterations'], 2)
        self.assertNotIn('direct_optimizer', effective)
        self.assertNotIn('shared_library', effective)
        self.assertEqual(config.deepmod_params, before)

    def test_variants_and_portable_manifest_keep_protocol(self):
        runs = []
        for method in methods.METHODS:
            for protocol in ('fixed', 'native'):
                cfg = load_config('burgers', f'{method}_pdecomp_{protocol}')
                self.assertEqual(cfg['method'], method)
                self.assertEqual(cfg[method]['protocol'], protocol)
            runs.append({'method': method, 'variant': f'{method}_pdecomp_native'})
        manifest = portable.make_manifest('burgers', 0, 3, runs)
        self.assertEqual([r['variant'] for r in manifest['runs']],
                         ['deepmod_pdecomp_native', 'discover_pdecomp_native'])

    def test_symbolic_equations_are_preserved_without_invented_metrics(self):
        raw = {'targets': ['u_t'], 'features': [['sin(u1)']], 'coefficients': [[0.5]]}
        with patch.object(methods, '_bridge', return_value={'result': raw, 'settings': {}, 'fit_seconds': 1}):
            front, objectives, _, info = methods.run(heat(), heat().data, 'discover')
        self.assertIn('sin(u1)', front[0][0])
        metrics = methods.score(front, objectives, [], info)
        self.assertNotIn('success_selected', metrics)
        result = contract.normalize({'status': 'ok', 'front': front, 'objectives': objectives,
                                     'metrics': metrics, 'selected': front[0]}, 'discover')
        self.assertEqual(result['selected'], front[0])
        self.assertTrue(metrics['scoring_note'])

    def test_source_manifest_is_pinned(self):
        manifest = json.loads(methods.MANIFEST.read_text())
        for method in methods.METHODS:
            self.assertTrue(any(row[2] == method + '/run.py' for row in manifest[method]))
            for repo, revision, name, destination, checksum in manifest[method]:
                self.assertEqual(len(revision), 40)
                self.assertEqual(len(checksum), 64)
                self.assertNotIn('..', Path(destination).parts)

    def test_uploaded_methods_receive_their_own_native_options(self):
        from support.discovery_lab import worker, job_queue
        from support.discovery_lab.lab_settings import _custom_base
        problem = heat()
        arrays = {'axis_0': problem.grids[0][:, 0], 'axis_1': problem.grids[1][0, :],
                  'axis_names': np.array(['t', 'x']), 'var_u': problem.data['u']}
        options = {'deepmod': {'max_iterations': 2}, 'discover': {'n_samples': 20, 'batch_size': 10}}
        settings = {'search': _custom_base(1), 'seed': 0, 'noise': 0, 'method_configs': {
            m: {'protocol': 'native', 'target_orders': {'u': 1}, 'overrides': opts}
            for m, opts in options.items()}}
        with tempfile.TemporaryDirectory() as tmp, patch.object(job_queue, 'ROOT', Path(tmp)):
            owner = job_queue.new_owner()
            data, config = job_queue.save_upload(owner, arrays, settings)
            folder = job_queue.job_path(owner, 'a' * 32)
            folder.mkdir(parents=True)
            for method in methods.METHODS:
                result = ([['1.0 * u{power: 1.0} = du/dx0{power: 1.0}']], [None], 1, {'protocol': 'native'})
                with patch.object(methods, 'run', return_value=result) as run:
                    record = worker._run_upload({'method': method, 'data_path': str(data),
                                                 'config_path': str(config)}, folder, hosted=True)
                self.assertEqual(record['status'], 'ok')
                self.assertEqual(run.call_args.kwargs['protocol'], 'native')
                self.assertEqual(run.call_args.kwargs['overrides'], options[method])

    def test_wizard_exposes_both_native_methods_and_keeps_settings(self):
        from streamlit.testing.v1 import AppTest
        defaults = {'max_iterations': 5000, 'n_samples': 5000, 'batch_size': 250}
        with patch.object(methods, 'describe', return_value=defaults):
            app = AppTest.from_file(str(PIC / 'app/Home.py'), default_timeout=40).run()
            app.switch_page('views/10_Discovery_lab.py').run()
            app.session_state['lab_discovery_methods'] = ['deepmod', 'discover']
            app.session_state['catalog_requested_record'] = 'nir1_heat'
            app.run()
            app.button(key='lab_next_data').click().run()
            self.assertFalse(app.exception)
            def control(label):
                return next(item for item in app.number_input if item.label == label)
            def budget():
                return next(item for item in app.get('button_group')
                            if item.key.startswith('lab_deepmod_nir1_heat_native_')
                            and item.key.endswith('_budget'))
            self.assertEqual(control('Learning rate').proto.format, '%g')
            self.assertEqual(control('Training iterations').value, 250)
            self.assertEqual(control('Maximum training samples').value, 1000)
            budget().set_value('PDEComp profile').run()
            self.assertEqual(control('Training iterations').value, 5000)
            budget().set_value('Quick demo').run()
            self.assertEqual(control('Training iterations').value, 250)
            control('Training iterations').set_value(25).run()
            app.button(key='lab_next_settings').click().run()
            self.assertFalse(app.exception)
            configuration = app.session_state['lab_draft_config']['extra_settings']
            self.assertEqual(configuration['deepmod']['protocol'], 'native')
            self.assertEqual(configuration['discover']['protocol'], 'native')
            self.assertEqual(configuration['deepmod']['overrides']['max_iterations'], 25)
            app.button(key='lab_back_settings').click().run()
            self.assertEqual(control('Training iterations').value, 25)


@unittest.skipUnless(os.environ.get('EPDE_TEST_NATIVE_METHODS') == '1', 'real NN/RNN smoke is opt-in')
class RealNativeCPU(unittest.TestCase):
    def test_native_deepmod_nn_autograd_training(self):
        problem = heat()
        front, objectives, seconds, info = methods.run(problem, problem.data, 'deepmod',
            overrides={'max_iterations': 2, 'write_iterations': 1, 'max_samples': 100,
                       'poly_order': 1, 'diff_order': 2})
        self.assertEqual(info['protocol'], 'native')
        self.assertEqual(info['settings']['max_iterations'], 2)
        self.assertEqual(len(front[0]), 1)
        self.assertEqual(objectives, [None])
        self.assertGreater(seconds, 0)
        self.assertTrue(np.isfinite(info['result']['coefficients']).all())

    def test_native_discover_controller_training(self):
        problem = heat()
        front, objectives, seconds, info = methods.run(problem, problem.data, 'discover',
            overrides={'n_samples': 20, 'batch_size': 10, 'max_length': 5, 'max_add_count': 2})
        self.assertEqual(info['protocol'], 'native')
        self.assertIsNone(info['library_size'])
        self.assertEqual(info['settings']['n_samples'], 20)
        self.assertIn('discover_reward', info['result'])
        self.assertGreater(seconds, 0)
        self.assertTrue(np.isfinite(info['result']['coefficients']).all())
        self.assertEqual(objectives, [None])
        self.assertTrue(front[0])

    def test_hosted_native_discover_ode_worker(self):
        from support.discovery_lab.worker import _run_record
        record = _run_record({'method': 'discover', 'source': 'record', 'dataset': 'ode',
                              'variant': 'discover_pdecomp_native', 'seed': 2, 'noise': 0,
                              'overrides': {'discover': {'overrides': {
                                  'n_samples': 20, 'batch_size': 10, 'max_length': 5, 'max_add_count': 2}}}}, hosted=True)
        self.assertEqual(record['status'], 'ok', record.get('error'))
        self.assertEqual(record['baseline']['protocol'], 'native')
        self.assertEqual(record['baseline']['settings']['seed'], 2)
        self.assertTrue(contract.normalize(record, 'discover')['selected'])


if __name__ == '__main__':
    unittest.main()
