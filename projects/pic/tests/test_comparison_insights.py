"""Comparison must reward measured, compatible outcomes rather than missing data."""
import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PIC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PIC), str(PIC / 'app'), str(PIC.parents[1])]


def record(method='epde', *, hd=0, re=0.1, r2=0.9, seconds=20):
    variant = 'default' if method == 'epde' else f'{method}_pdecomp_native'
    return {
        'method': method, 'variant': variant, 'status': 'ok',
        'dataset': 'burgers', 'noise': 0, 'seed': 0,
        'config': {'method': method, 'loader': {}, method: {'protocol': 'native'}},
        'identity': {'source_sha256': 'a' * 64, 'data_sha256': 'b' * 64,
                     'job': {'dataset': 'burgers', 'noise': 0, 'seed': 0}},
        'problem': {'kind': 'pde_1d', 'variables': ['u'], 'truth': ['known reference']},
        'selected': ['selected equation'], 'front': [['selected equation']],
        'metrics': {'hamming_selected': hd, 're_sum': re,
                    'success_selected': hd == 0, 'success_front': hd == 0},
        'baseline': {} if method == 'epde' else {'protocol': 'native'},
        'fit_seconds': seconds,
        'environment': {'platform': 'Linux-test', 'machine': 'x86_64', 'cpu_count': 4,
                        'python': '3.12.8', 'blas_threads': '1', 'cuda': False,
                        'numpy': '2.3.4', 'scipy': '1.16.2', 'torch': '2.9.0',
                        'pysindy': '2.1.0'},
        'evaluation': {'kind': 'pde_1d', 'variables': ['u'], 'axis_names': ['t', 'x'],
                       'solution': {'success': True, 'coverage': 1.0,
                                    'r2': {'u': r2}, 'nrmse': {'u': 0.2},
                                    't': [0, 1, 2], 'x': [0, 1],
                                    'observed': {'u': [[1, 2], [2, 3], [3, 4]]}}},
    }


class ComparisonInsights(unittest.TestCase):
    def analyze(self, runs):
        # A missing implementation is a clear feature failure, not an import error.
        import importlib.util
        self.assertIsNotNone(importlib.util.find_spec('support.discovery_lab.comparison_insights'),
                             'The comparison analysis module has not been implemented')
        from support.discovery_lab.comparison_insights import analyze_comparison
        return analyze_comparison(runs)

    @staticmethod
    def metric(analysis, key):
        return next(item for item in analysis['metrics'] if item['key'] == key)

    def test_four_methods_get_metric_specific_leaders_without_an_overall_winner(self):
        runs = [('EPDE', record(hd=1, re=0.3, r2=0.99, seconds=30)),
                ('PySINDy', record('pdecomp', hd=0, re=0.05, r2=0.98, seconds=2)),
                ('DeepMoD', record('deepmod', hd=0, re=0.08, r2=0.95, seconds=15)),
                ('DISCOVER', record('discover', hd=2, re=0.5, r2=0.8, seconds=25))]
        before = copy.deepcopy(runs)
        result = self.analyze(runs)
        self.assertTrue(result['verified_input'])
        self.assertEqual(self.metric(result, 'hd')['leaders'], ['PySINDy', 'DeepMoD'])
        self.assertEqual(self.metric(result, 're')['leaders'], ['PySINDy'])
        self.assertEqual(self.metric(result, 'solution_r2')['leaders'], ['EPDE'])
        self.assertEqual(self.metric(result, 'seconds')['leaders'], ['PySINDy'])
        self.assertNotIn('winner', result)
        self.assertEqual(runs, before)
        self.assertTrue(any('held-out' in note for note in result['notes']))

    def test_unscorable_and_missing_metrics_never_become_zero_or_leaders(self):
        unavailable = record('discover', hd=0, re=0, r2=1, seconds=1)
        unavailable['metrics']['scoring_note'] = 'Expression outside shared grammar'
        unavailable['evaluation'] = {'solution_note': 'Expression cannot be reconstructed'}
        missing = record('deepmod')
        missing['metrics'] = {}
        missing['evaluation'] = {}
        result = self.analyze([('EPDE', record()), ('DISCOVER', unavailable), ('DeepMoD', missing)])
        self.assertEqual(self.metric(result, 'hd')['eligible'], 1)
        self.assertEqual(self.metric(result, 'hd')['leaders'], [])
        self.assertIsNone(result['runs'][1]['hd'])
        self.assertIsNone(result['runs'][2]['hd'])
        self.assertEqual(result['runs'][2]['selected_law'], 'Not scored')

    def test_oracle_candidate_error_does_not_replace_selected_equation_error(self):
        candidate = record(hd=2, re=None)
        candidate['metrics'].update(success_front=True, best_index=0, selected_index=1,
                                    coef_error=0.0, hamming_min=0)
        candidate['front'] = [['oracle candidate'], ['selected equation']]
        result = self.analyze([('EPDE', candidate), ('PySINDy', record('pdecomp', re=0.2))])
        self.assertIsNone(result['runs'][0]['re'])
        self.assertEqual(self.metric(result, 're')['leaders'], [])
        self.assertTrue(any('selection' in tip.lower() for tip in result['runs'][0]['advice']))
        self.assertEqual(result['runs'][0]['selected_law'], 'Different structure')

    def test_mismatched_or_missing_input_provenance_prevents_all_leaders(self):
        for change in ('source', 'data', 'missing'):
            with self.subTest(change=change):
                other = record('pdecomp')
                if change == 'missing':
                    other.pop('identity')
                else:
                    other['identity'][change + '_sha256'] = 'c' * 64
                result = self.analyze([('EPDE', record()), ('PySINDy', other)])
                self.assertFalse(result['verified_input'])
                self.assertTrue(all(not m['leaders'] for m in result['metrics']))

    def test_different_environment_blocks_only_time_leader(self):
        other = record('pdecomp', re=0.01, seconds=1)
        other['environment']['machine'] = 'arm64'
        result = self.analyze([('EPDE', record()), ('PySINDy', other)])
        self.assertEqual(self.metric(result, 're')['leaders'], ['PySINDy'])
        self.assertEqual(self.metric(result, 'seconds')['leaders'], [])
        self.assertIn('environment', self.metric(result, 'seconds')['explanation'].lower())

    def test_incomplete_solution_or_changed_observation_grid_has_no_fit_leader(self):
        for change in ('coverage', 'grid', 'missing_variable'):
            with self.subTest(change=change):
                other = record('deepmod', r2=1.0)
                if change == 'coverage':
                    other['evaluation']['solution']['coverage'] = 0.5
                elif change == 'grid':
                    other['evaluation']['solution']['t'] = [0, 0.5, 1]
                else:
                    other['problem']['variables'] = ['u', 'v']
                    other['evaluation']['variables'] = ['u', 'v']
                result = self.analyze([('EPDE', record()), ('DeepMoD', other)])
                self.assertEqual(self.metric(result, 'solution_r2')['leaders'], [])

    def test_timeout_is_not_fastest_and_has_actionable_budget_advice(self):
        other = record('deepmod', seconds=0.01)
        other['status'] = 'timeout'
        result = self.analyze([('EPDE', record()), ('DeepMoD', other)])
        self.assertIsNone(result['runs'][1]['seconds'])
        self.assertEqual(self.metric(result, 'seconds')['leaders'], [])
        self.assertTrue(any('budget' in tip.lower() for tip in result['runs'][1]['advice']))

    def test_legacy_record_and_nonfinite_metrics_remain_unavailable(self):
        old = record('pdecomp')
        old.pop('environment')
        old['metrics'] = {'success_front': True, 'coef_error': 0.0,
                          'hamming_selected': float('nan'), 're_sum': float('inf')}
        old['evaluation']['solution']['r2']['u'] = None
        old['fit_seconds'] = True
        result = self.analyze([('EPDE', record()), ('legacy', old)])
        for key in ('hd', 're', 'solution_r2', 'seconds'):
            self.assertIsNone(result['runs'][1][key])
            self.assertEqual(self.metric(result, key)['leaders'], [])

    def test_protocol_difference_is_explained_without_comparing_internal_objectives(self):
        other = record('pdecomp', re=0.01)
        other['baseline']['protocol'] = 'fixed'
        other['objectives'] = [[-1e100, -1e100]]
        first = record()
        first['objectives'] = [[1e100, 1e100]]
        result = self.analyze([('EPDE', first), ('PySINDy fixed', other)])
        self.assertEqual(self.metric(result, 're')['leaders'], ['PySINDy fixed'])
        self.assertTrue(any('protocol' in note.lower() for note in result['notes']))
        self.assertFalse(any(m['key'] in ('objective', 'reward') for m in result['metrics']))

    def test_different_reference_laws_block_structural_ranking(self):
        other = record('pdecomp', hd=0)
        other['problem']['truth'] = ['different reference']
        result = self.analyze([('EPDE', record(hd=2)), ('PySINDy', other)])
        self.assertEqual(self.metric(result, 'hd')['leaders'], [])
        self.assertEqual(self.metric(result, 're')['leaders'], [])

    def test_explicit_matching_target_is_compatible_but_changed_target_is_not(self):
        first, other = record(), record('deepmod', re=0.01)
        for rec in (first, other):
            rec['dataset'] = rec['identity']['job']['dataset'] = 'custom'
            rec['problem']['truth'] = ['-1.0 * u{power: 1.0} = du/dx0{power: 1.0}']
        first['config'].pop('epde')
        other['config']['deepmod']['target_orders'] = {'u': 1}
        result = self.analyze([('EPDE', first), ('DeepMoD', other)])
        self.assertEqual(self.metric(result, 're')['leaders'], ['DeepMoD'])
        other['config']['deepmod']['target_orders']['u'] = 2
        result = self.analyze([('EPDE', first), ('DeepMoD', other)])
        self.assertEqual(self.metric(result, 'hd')['leaders'], [])
        self.assertEqual(self.metric(result, 're')['leaders'], [])

    def test_environment_timestamps_do_not_block_observed_time_and_different_budgets_are_explicit(self):
        first, other = record(), record(seconds=10)
        first['environment']['captured_at'] = '2026-01-01'
        other['environment']['captured_at'] = '2026-01-02'
        first['search_config'] = {'epochs': 5}
        other['search_config'] = {'epochs': 10}
        result = self.analyze([('short budget', first), ('long budget', other)])
        self.assertEqual(self.metric(result, 'seconds')['leaders'], ['long budget'])
        self.assertTrue(any('configurations differ' in note for note in result['notes']))
        self.assertIn('observed time', self.metric(result, 'seconds')['explanation'])

    def test_forecast_requires_explicit_successful_matching_holdout(self):
        def forecast(rec, score):
            rec['protocol'] = 'chronological EPDE discovery and frozen-equation forecast'
            rec['train'] = {'data_sha256': 'd' * 64, 'n_points': 7}
            rec['test'] = {'n_points': 3, 'time_range': [7, 9], 'integration_success': True,
                           'coverage': 1, 'metrics': {'u': {'r2': score}}}
            rec['split_index'], rec['total_points'] = 7, 10
            return rec
        first, other = forecast(record(), 0.8), forecast(record('pdecomp'), 0.9)
        result = self.analyze([('EPDE', first), ('PySINDy', other)])
        self.assertEqual(self.metric(result, 'heldout_r2')['leaders'], ['PySINDy'])
        other['test']['time_range'] = [8, 10]
        result = self.analyze([('EPDE', first), ('PySINDy', other)])
        self.assertEqual(self.metric(result, 'heldout_r2')['leaders'], [])


class ComparisonPage(unittest.TestCase):
    def test_equal_length_ode_grids_use_their_own_observations_for_residuals(self):
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError:
            self.skipTest('streamlit is not installed')
        from support.discovery_lab import job_queue, ui
        import matplotlib.pyplot as plt

        saved = {'full': record(), 'short': record('deepmod')}
        equation = '1.0 * u{power: 1.0} + 0.0 = du/dx0{power: 1.0}'
        for identifier, rec in saved.items():
            rec['selected'], rec['front'] = [equation], [[equation]]
            rec['problem'].update(truth=[equation], kind='ode')
            rec['evaluation']['kind'] = 'ode'
            solution = rec['evaluation']['solution']
            solution.pop('x')
            solution['t'] = [0, 1, 2] if identifier == 'full' else [0, 0.5, 1]
            solution['observed'] = {'u': [1, 2, 3] if identifier == 'full' else [1, 1.5, 2]}
            solution['simulated'] = copy.deepcopy(solution['observed'])
        jobs = [{'id': identifier, 'method': rec['method'], 'variant': rec['variant'],
                 'state': 'finished', 'dataset': 'burgers', 'noise': 0, 'seed': 0}
                for identifier, rec in saved.items()]
        figures = []
        with patch.object(job_queue, 'list_jobs', return_value=jobs), \
             patch.object(job_queue, 'result', side_effect=lambda owner, identifier: saved[identifier]), \
             patch.object(ui, 'show_figure', side_effect=figures.append):
            app = AppTest.from_file(str(PIC / 'app' / 'Home.py'), default_timeout=60).run()
            app.switch_page('views/12_Discovery_compare.py').run()
        try:
            self.assertFalse(app.exception, [item.value for item in app.exception])
            residuals = [line.get_ydata() for fig in figures for ax in fig.axes
                         if ax.get_ylabel() == 'error, u' for line in ax.lines
                         if line.get_label().startswith(('EPDE', 'DeepMoD'))]
            self.assertEqual(len(residuals), 2)
            for residual in residuals:
                self.assertEqual(list(residual), [0, 0, 0])
        finally:
            for fig in figures:
                plt.close(fig)

    def test_same_input_with_different_solution_grids_renders_without_overlaying_residuals(self):
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError:
            self.skipTest('streamlit is not installed')
        from support.discovery_lab import job_queue

        saved = {'full': record(), 'short': record('deepmod')}
        equation = '1.0 * u{power: 1.0} + 0.0 = du/dx0{power: 1.0}'
        for rec in saved.values():
            rec['selected'], rec['front'] = [equation], [[equation]]
            rec['problem']['truth'] = [equation]
        short = saved['short']['evaluation']['solution']
        short['t'], short['observed']['u'] = [0, 2], [[1, 2], [3, 4]]
        for rec in saved.values():
            solution = rec['evaluation']['solution']
            solution['simulated'] = copy.deepcopy(solution['observed'])
        jobs = [{'id': identifier, 'method': rec['method'], 'variant': rec['variant'],
                 'state': 'finished', 'dataset': 'burgers', 'noise': 0, 'seed': 0}
                for identifier, rec in saved.items()]
        with patch.object(job_queue, 'list_jobs', return_value=jobs), \
             patch.object(job_queue, 'result', side_effect=lambda owner, identifier: saved[identifier]):
            app = AppTest.from_file(str(PIC / 'app' / 'Home.py'), default_timeout=60).run()
            app.switch_page('views/12_Discovery_compare.py').run()
        self.assertFalse(app.exception, [item.value for item in app.exception])
        fit = next(metric for metric in app.metric if metric.label == 'In-sample solution R²')
        self.assertEqual(fit.value, '—')
        self.assertTrue(any('own recorded grid' in caption.value for caption in app.caption))

    def test_four_distinct_methods_default_to_insights_with_honest_missing_metrics(self):
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError:
            self.skipTest('streamlit is not installed')
        from support.discovery_lab import job_queue

        methods = ['epde', 'epde', 'pdecomp', 'deepmod', 'discover']
        saved = {}
        jobs = []
        equation = '1.0 * u{power: 1.0} + 0.0 = du/dx0{power: 1.0}'
        for index, method in enumerate(methods):
            identifier = f'comparison-{index}'
            rec = record(method, hd=index % 2, re=0.1 * index, seconds=index + 1)
            rec['selected'], rec['front'] = [equation], [[equation]]
            rec['problem']['truth'] = [equation]
            rec['evaluation'] = {'solution_note': 'No solution reconstruction stored for this test.'}
            if method == 'discover':
                rec['metrics'] = {'scoring_note': 'Native expression is outside shared scoring.'}
            saved[identifier] = rec
            jobs.append({'id': identifier, 'method': method, 'variant': rec['variant'],
                         'state': 'finished', 'dataset': 'burgers', 'noise': 0, 'seed': 0,
                         'created': index, 'effort': 'quick'})
        with patch.object(job_queue, 'list_jobs', return_value=jobs), \
             patch.object(job_queue, 'result', side_effect=lambda owner, identifier: saved[identifier]):
            app = AppTest.from_file(str(PIC / 'app' / 'Home.py'), default_timeout=60).run()
            app.switch_page('views/12_Discovery_compare.py').run()
        self.assertFalse(app.exception, [item.value for item in app.exception])
        self.assertEqual([tab.label for tab in app.tabs],
                         ['Insights', 'Equations', 'Metrics', 'Solution fit', 'Equation balance'])
        self.assertEqual(app.multiselect[0].value,
                         ['comparison-0', 'comparison-2', 'comparison-3', 'comparison-4'])
        headings = [item.value for item in app.subheader]
        self.assertIn('What these runs show', headings)
        self.assertIn('What to improve next', headings)
        markdown = '\n'.join(item.value for item in app.markdown)
        for method in ('EPDE', 'PySINDy', 'DeepMoD', 'DISCOVER'):
            self.assertIn(method, markdown)
        self.assertIn('shared structural scoring', markdown)
        frame = app.dataframe[0].value
        self.assertNotIn('coefficient error', frame.columns)
        discover = frame[frame['method'].str.startswith('DISCOVER')].iloc[0]
        self.assertEqual(discover['selected equation'], 'Not scored')
        self.assertTrue(__import__('pandas').isna(discover['HD']))


class ActualBudgetLabels(unittest.TestCase):
    def test_effective_epde_budget_overrides_the_generic_effort_name(self):
        from support.discovery_lab.ui import budget_label
        saved = {'method': 'epde', 'effort': 'standard', 'search_config': {
            'evolution': {'population_size': 4, 'training_epochs': 1}}}
        self.assertEqual(budget_label({}, saved), '4 candidates × 1 epoch')

    def test_queued_custom_budget_is_not_mislabelled_standard(self):
        from support.discovery_lab.ui import budget_label
        self.assertEqual(budget_label({'method': 'epde', 'effort': 'custom'}), 'custom budget')


if __name__ == '__main__':
    unittest.main()
