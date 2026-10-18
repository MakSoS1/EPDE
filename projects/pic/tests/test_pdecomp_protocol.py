"""PySINDy under the PDEComp protocols, PDEComp's metrics and series summaries.

The conversion and metric tests need no network. The parity tests run
PDEComp's own code (downloaded once from the pinned commit, or taken from
PDECOMP_DIR) and are skipped when it cannot be obtained.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

PIC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PIC / 'app'), str(PIC), str(PIC.parents[1])]

from epde_bench import metrics, pdecomp                       # noqa: E402
from epde_bench.datasets import load                          # noqa: E402


def _hd(found, problem):
    """Hamming distance of a found system to the record's accepted laws."""
    return metrics.hamming_best(metrics.canonical_tokens(found),
                                [metrics.canonical_tokens(t) for t in problem.truth_systems])


def _pdecomp_available():
    try:
        pdecomp.source_dir()
        return True
    except pdecomp.PdecompUnavailable:
        return False


class Conversion(unittest.TestCase):
    def test_product_token_of_the_record(self):
        problem = load('kdv_cossin')
        raw = {'targets': ['u_t'], 'features': [['u u_x', 'u_xxx', 'sin(x) cos(t)', 'u']],
               'coefficients': [[-6.0, -1.0, 1.0, 0.0]]}
        self.assertEqual(_hd(pdecomp.to_epde_texts(problem, raw), problem), 0)

    def test_compound_term_is_expanded(self):
        problem = load('pde_compound')
        raw = {'targets': ['u_t'], 'features': [['d_x(u u_x)']], 'coefficients': [[1.0]]}
        self.assertEqual(_hd(pdecomp.to_epde_texts(problem, raw), problem), 0)

    def test_system_variables_and_reciprocal(self):
        lv = load('pdecomp_lv')
        raw = {'targets': ['x0_t', 'x1_t'], 'features': [['x0', 'x0 x1'], ['x1', 'x0 x1']],
               'coefficients': [[20.0, -20.0], [-20.0, 20.0]]}
        self.assertEqual(_hd(pdecomp.to_epde_texts(lv, raw), lv), 0)
        divide = load('pdecomp_pde_divide')
        raw = {'targets': ['u_t'], 'features': [['u_xx', '(1/x) u_x', '1']],
               'coefficients': [[0.25, -1.0, 0.0]]}
        self.assertEqual(_hd(pdecomp.to_epde_texts(divide, raw), divide), 0)

    def test_trigonometric_and_constant(self):
        problem = load('pdecomp_ode_simple')
        raw = {'targets': ['u_t'], 'features': [['cos(t)', 'sin(t)', '1']],
               'coefficients': [[1.0, -1.3, 0.5]]}
        (eq,) = pdecomp.to_epde_texts(problem, raw)
        self.assertIn('0.5 = ', eq)
        self.assertEqual(_hd([eq], problem), 1)          # the free coefficient is an extra term


class RelativeErrorSum(unittest.TestCase):
    truth = [['-4.0 * u{power: 1.0} + 1.5 * x{power: 1.0, dim: 0.0} = d^2u/dx0^2{power: 1.0}']]

    def test_exact_and_scaled(self):
        self.assertAlmostEqual(metrics.relative_error_sum(self.truth[0], self.truth), 0.0)
        doubled = ['-8.0 * u{power: 1.0} + 3.0 * x{power: 1.0, dim: 0.0} = 2.0 * d^2u/dx0^2{power: 1.0}']
        self.assertAlmostEqual(metrics.relative_error_sum(doubled, self.truth), 0.0)

    def test_errors_and_missing_terms(self):
        found = ['-4.4 * u{power: 1.0} + 0.1 * u{power: 2.0} = d^2u/dx0^2{power: 1.0}']
        # 10 % on u, the t term missing (adds 1), the extra term adds nothing
        self.assertAlmostEqual(metrics.relative_error_sum(found, self.truth), 1.1)

    def test_free_coefficient_counts_as_a_term(self):
        with_constant = ['-4.0 * u{power: 1.0} + 1.5 * x{power: 1.0, dim: 0.0} + 0.18 = d^2u/dx0^2{power: 1.0}']
        with_zero = ['-4.0 * u{power: 1.0} + 1.5 * x{power: 1.0, dim: 0.0} + 0.0 = d^2u/dx0^2{power: 1.0}']
        truth = [metrics.canonical_tokens(t) for t in self.truth]
        self.assertEqual(metrics.hamming_best(metrics.canonical_tokens(with_constant), truth), 1)
        self.assertEqual(metrics.hamming_best(metrics.canonical_tokens(with_zero), truth), 0)

    def test_single_candidate_is_selected(self):
        out = metrics.score_run([self.truth[0]], [None], self.truth)
        self.assertEqual(out['selected_index'], 0)
        self.assertTrue(out['success_selected'])
        self.assertAlmostEqual(out['re_sum'], 0.0)


class SettingsChanges(unittest.TestCase):
    def test_validation(self):
        ok = pdecomp.validate_overrides({'optimizer': {'type': 'FROLS', 'max_iter': 3}, 'crop': 5}, 'fixed')
        self.assertEqual(ok, {'optimizer': {'type': 'FROLS', 'max_iter': 3}, 'crop': 5})
        for bad, protocol in (({'optimizer': {'type': 'SR3'}}, 'fixed'),
                              ({'crop': 5}, 'native'),
                              ({'polynomial_degree': 3}, 'fixed'),
                              ({'optimizer': {'threshold': -1}}, 'fixed'),
                              ({'differentiation': {'order': 3}}, 'native'),
                              ({'library': {'custom_tokens': ['exp(u)']}}, 'fixed'),
                              ({'library': {'equation_factors_max_number': 3}}, 'fixed'),
                              ({'library': {'max_deriv_order': [3, 4]}}, 'fixed'),
                              ({'differentiation': {'periodic': 'yes'}}, 'native')):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                pdecomp.validate_overrides(bad, protocol)


class Series(unittest.TestCase):
    def _record(self, noise, ok):
        return {'status': 'ok', 'noise': noise, 'fit_seconds': 1.0, 'problem': {'truth': ['x']},
                'metrics': {'success_selected': ok, 'hamming_selected': 0 if ok else 2,
                            're_sum': 0.01 if ok else 1.0}}

    def test_noise_limit_follows_three_of_five(self):
        from support.discovery_lab import series
        pairs = []
        for noise, wins in ((0.0, 5), (1.0, 3), (5.0, 2)):
            for k in range(5):
                pairs.append(({'method': 'pdecomp'}, self._record(noise, k < wins)))
        rows, limits = series.summarize(pairs, lambda job: 'M')
        self.assertEqual([r['found the law'] for r in rows], [5, 3, 2])
        self.assertEqual(limits, {'M': 1.0})
        self.assertAlmostEqual(rows[1]['RE (sum), mean of successes'], 0.01)

    def test_errors_count_as_failures(self):
        from support.discovery_lab import series
        pairs = [({}, self._record(0.0, True))] * 2 + [({}, {'status': 'error', 'noise': 0.0,
                                                             'problem': {'truth': ['x']}})]
        rows, limits = series.summarize(pairs, lambda job: 'M')
        self.assertEqual((rows[0]['runs'], rows[0]['found the law'], rows[0]['failed runs']), (3, 2, 1))
        self.assertEqual(limits, {'M': 0.0})


class Records(unittest.TestCase):
    def test_windows_match_pdecomp(self):
        self.assertEqual(load('pdecomp_lv').shape, (150,))
        self.assertEqual(load('pdecomp_lorenz').shape, (1000,))
        divide = load('pdecomp_pde_divide')
        self.assertAlmostEqual(float(np.asarray(divide.grids[0]).max()), 1.0)
        self.assertEqual(load('pdecomp_ode_simple').shape, (200,))
        self.assertEqual(set(pdecomp.PDECOMP_FILES.values()) - {'ns_data.mat'},
                         {'ode_data.npy', 'vdp_data.npy', 'lotka_data.npy', 'lorenz_data.npy',
                          'burgers_data.mat', 'burgers_sln_100_data.csv', 'ac_data.npy',
                          'kdv_data.mat', 'kdv_periodic_data.npy', 'wave_data.csv',
                          'pde_divide_data.npy', 'pde_compound_data.npy', 'ks_data.mat',
                          'ODE_simple_discovery'})


@unittest.skipUnless(_pdecomp_available(), 'PDEComp source is not reachable')
class Parity(unittest.TestCase):
    """The same numbers as PDEComp's own scorer, where both write the law alike."""

    def test_fixed_and_native(self):
        from epde_bench.runner import run_one
        for dataset in ('ode', 'pdecomp_lv', 'pdecomp_pde_divide', 'pdecomp_ode_simple', 'burgers'):
            for protocol in ('fixed', 'native'):
                with self.subTest(dataset=dataset, protocol=protocol):
                    record = run_one(dataset, f'pysindy_pdecomp_{protocol}', 0.0, 0)
                    self.assertEqual(record['status'], 'ok', record.get('error'))
                    own = record['baseline']['pdecomp_metrics']
                    self.assertEqual(record['metrics']['hamming_selected'], own['hd'])
                    self.assertAlmostEqual(record['metrics']['re_sum'], own['re'], places=6)
                    self.assertTrue(record['metrics']['success_selected'])

    def test_changed_settings_reach_pdecomp(self):
        from epde_bench.runner import run_one
        base = run_one('ode', 'pysindy_pdecomp_fixed', 0.0, 0)
        self.assertEqual(base['baseline']['overrides'], {})
        changed = run_one('ode', 'pysindy_pdecomp_fixed', 0.0, 0,
                          overrides={'pdecomp': {'overrides': {'optimizer': {'type': 'FROLS', 'max_iter': 3}}}})
        self.assertEqual(changed['baseline']['settings']['optimizer']['type'], 'FROLS')
        self.assertIn('changed by the user', changed['baseline']['pdecomp_settings'])
        native = pdecomp.describe(None, 'native', {'polynomial_degree': 2})
        self.assertEqual(native['polynomial_degree'], 2)
        self.assertEqual(pdecomp.describe('ode', 'fixed')['optimizer']['threshold'], 1e-06)

    def test_library_settings_reach_pdecomp(self):
        from epde_bench.runner import run_one
        sizes = {}
        for name, change in (('base', None), ('single', {'equation_factors_max_number': 1}),
                             ('no_bias', {'include_bias': False})):
            record = run_one('ode', 'pysindy_pdecomp_fixed', 0.0, 0,
                             overrides={'pdecomp': {'overrides': {'library': change}}} if change else None)
            sizes[name] = sum(record['baseline']['library_size'].values())
        self.assertEqual(sizes, {'base': 12, 'single': 9, 'no_bias': 11})
        with self.assertRaises(ValueError):
            pdecomp.validate_overrides({'library': {'custom_tokens': ['sin(x)']}}, 'fixed', 'ode')

    def test_unsupported_record(self):
        from epde_bench.runner import run_one
        record = run_one('duffing', 'pysindy_pdecomp_fixed', 0.0, 0)
        self.assertEqual(record['status'], 'unsupported')

    def test_noisy_runs_agree_with_pdecomp_including_free_coefficient(self):
        from epde_bench.runner import run_one
        for seed in range(3):
            record = run_one('ode', 'pysindy_pdecomp_fixed', 1.0, seed)
            own = record['baseline']['pdecomp_metrics']
            self.assertEqual(record['metrics']['hamming_selected'], own['hd'])

    def test_worker_evaluates_the_result(self):
        from epde_bench.runner import run_one
        from support.discovery_lab.worker import evaluate
        record = run_one('pdecomp_pde_divide', 'pysindy_pdecomp_fixed', 0.0, 0)
        evaluation = evaluate(record)
        (balance,) = evaluation['balance'].values()
        self.assertGreater(balance, 0.9)
        self.assertIn('observed', next(iter(evaluation['balance_series'].values())))


if __name__ == '__main__':
    unittest.main()
