"""Scientific result and reproducible-export regressions from PR review."""
import io
import os
import subprocess
import sys
import tempfile
import tomllib
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

PIC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PIC), str(PIC / 'app'), str(PIC.parents[1])]

from epde_bench import pdecomp_methods as methods
from epde_bench.problem import Problem
from support.discovery_lab import portable, series


class LockedDependencies(unittest.TestCase):
    def test_cpu_and_cuda_match_cloud_torch(self):
        project = tomllib.loads((PIC / 'pyproject.toml').read_text())
        for extra in ('cpu', 'cu128'):
            self.assertEqual(project['project']['optional-dependencies'][extra], ['torch==2.9.0'])
        lock = tomllib.loads((PIC / 'uv.lock').read_text())
        versions = [p['version'] for p in lock['package'] if p['name'] == 'torch']
        self.assertTrue(versions)
        self.assertTrue(all(v.split('+')[0] == '2.9.0' for v in versions), versions)


class RepositoryExport(unittest.TestCase):
    def test_heavy_export_button_uses_the_same_revision_as_the_experiment(self):
        from streamlit.testing.v1 import AppTest
        source = {'url': 'https://github.com/emil817/EPDE_demo', 'revision': 'c' * 40,
                  'archive_url': 'https://github.com/emil817/EPDE_demo/archive/' + 'c' * 40 + '.zip'}
        with patch.object(portable, 'repository_source', return_value=source), \
                patch.object(portable, 'make_manifest', wraps=portable.make_manifest) as make:
            app = AppTest.from_file(str(PIC / 'app/Home.py'), default_timeout=60).run()
            app.switch_page('views/10_Discovery_lab.py').run()
            app.session_state['lab_discovery_methods'] = ['epde']
            app.session_state['catalog_requested_record'] = 'ns'
            app.run()
            app.button(key='lab_next_data').click().run()
            app.button(key='lab_next_settings').click().run()
            app.checkbox(key='lab_confirm_science').check().run()
        self.assertFalse(app.exception)
        link = next(b for b in app.get('link_button') if b.proto.label == 'Download EPDE repository ZIP')
        self.assertEqual(link.proto.url, source['archive_url'])
        self.assertEqual(make.call_args.kwargs['repository'], source)
        with patch.object(portable, 'repository_source', side_effect=ValueError('dirty checkout')):
            app.run()
        self.assertFalse(app.exception)
        self.assertFalse(app.get('link_button'))
        self.assertFalse(app.get('download_button'))

    def test_archive_and_experiment_use_checkout_sha_not_demo_branch(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / '.git').mkdir()
            revision = 'a' * 40
            outputs = [subprocess.CompletedProcess([], 0, revision),
                       subprocess.CompletedProcess([], 0, '')]
            with patch('epde_bench.paths.REPO_ROOT', root), patch.dict(os.environ, {
                    'EPDE_REPOSITORY_REVISION': '', 'EPDE_REPOSITORY_URL': 'https://github.com/emil817/EPDE_demo'}), \
                    patch.object(portable.subprocess, 'run', side_effect=outputs):
                source = portable.repository_source()
            self.assertEqual(source['archive_url'], f'https://github.com/emil817/EPDE_demo/archive/{revision}.zip')
            manifest = portable.make_manifest('burgers', 0, 0, [
                {'method': 'deepmod', 'variant': 'deepmod_pdecomp_native'},
                {'method': 'discover', 'variant': 'discover_pdecomp_native'},
            ], repository=source)
            with zipfile.ZipFile(io.BytesIO(portable.bundle_bytes(manifest))) as z:
                self.assertIn(source['archive_url'], z.read('README.txt').decode())
            bundle = root / 'experiment.zip'
            bundle.write_bytes(portable.bundle_bytes(manifest))
            self.assertEqual(portable.read_manifest(bundle)['repository'], source)

    def test_dirty_checkout_never_links_to_unmodified_commit(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / '.git').write_text('gitdir: worktree')
            outputs = [subprocess.CompletedProcess([], 0, 'a' * 40),
                       subprocess.CompletedProcess([], 0, ' M projects/pic/epde_bench/pdecomp_methods.py')]
            with patch('epde_bench.paths.REPO_ROOT', root), \
                    patch.object(portable.subprocess, 'run', side_effect=outputs), \
                    self.assertRaisesRegex(ValueError, 'clean committed'):
                portable.repository_source()

    def test_gitless_build_needs_explicit_immutable_revision(self):
        with tempfile.TemporaryDirectory() as td, patch('epde_bench.paths.REPO_ROOT', Path(td)):
            for revision in ('', 'demo/streamlit-community-cloud', 'HEAD', '7fd9ece'):
                with patch.dict(os.environ, {'EPDE_REPOSITORY_REVISION': revision}), \
                        self.assertRaises(ValueError):
                    portable.repository_source()
            with patch.dict(os.environ, {'EPDE_REPOSITORY_REVISION': 'b' * 40}):
                self.assertEqual(portable.repository_source()['revision'], 'b' * 40)

    def test_build_sha_cannot_disagree_with_checkout(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / '.git').mkdir()
            outputs = [subprocess.CompletedProcess([], 0, 'a' * 40),
                       subprocess.CompletedProcess([], 0, '')]
            with patch('epde_bench.paths.REPO_ROOT', root), \
                    patch.object(portable.subprocess, 'run', side_effect=outputs), \
                    patch.dict(os.environ, {'EPDE_REPOSITORY_REVISION': 'b' * 40}), \
                    self.assertRaisesRegex(ValueError, 'differs'):
                portable.repository_source()


class IncompleteSeries(unittest.TestCase):
    def record(self, noise=0, metrics=None, status='ok'):
        return {'status': status, 'method': 'discover', 'noise': noise, 'problem': {'truth': ['u_t = u']},
                'metrics': {'success_selected': True, 'hamming_selected': 0} if metrics is None else metrics}

    def summarize(self, records):
        return series.summarize([({}, r) for r in records], lambda job: 'DISCOVER')

    def test_three_successes_and_two_unscored_keep_five_runs(self):
        records = [self.record() for _ in range(3)] + [
            self.record(metrics={'scoring_note': 'unsupported expression'}) for _ in range(2)]
        rows, limits = self.summarize(records)
        row = rows[0]
        self.assertEqual((row['runs'], row['scored runs'], row['unscored runs'], row['found the law']), (5, 3, 2, 3))
        self.assertIsNone(row['success rate'])
        self.assertIsNone(row['95 % interval'])
        self.assertEqual(limits, {'DISCOVER': None})

    def test_unscored_noise_level_is_not_skipped_to_certify_later_level(self):
        records = [self.record(noise) for noise in (0, 5) for _ in range(5)] + [
            self.record(1, {'scoring_note': 'unavailable'}) for _ in range(5)]
        rows, limits = self.summarize(records)
        self.assertEqual([r['noise, %'] for r in rows], [0, 1, 5])
        self.assertEqual(rows[1]['runs'], 5)
        self.assertEqual(rows[1]['unscored runs'], 5)
        self.assertEqual(limits, {'DISCOVER': None})

    def test_missing_score_and_baseline_note_are_also_unscored(self):
        record = self.record()
        record['baseline'] = {'scoring_note': 'no scoring'}
        rows, _ = self.summarize([record, self.record(metrics={})])
        self.assertEqual(rows[0]['unscored runs'], 2)
        self.assertEqual(rows[0]['found the law'], 0)

    def test_errors_remain_failures_and_other_methods_remain_complete(self):
        records = [self.record() for _ in range(3)] + [self.record(status=s) for s in ('error', 'timeout')]
        rows, limits = self.summarize(records)
        self.assertEqual(rows[0]['runs'], 5)
        self.assertEqual(rows[0]['failed runs'], 2)
        self.assertEqual(rows[0]['unscored runs'], 0)
        self.assertEqual(rows[0]['success rate'], 0.6)
        self.assertEqual(limits, {'DISCOVER': 0})

    def test_compare_displays_incomplete_statistics_without_an_exception(self):
        from streamlit.testing.v1 import AppTest
        from support.discovery_lab import job_queue, ui
        jobs = [{'id': str(i), 'state': 'finished', 'method': 'discover',
                 'dataset': 'ode', 'seed': i, 'noise': noise}
                for i, noise in enumerate((0, 0, 0, 1, 1))]
        records = {j['id']: self.record(j['noise'], {} if j['noise'] else None) for j in jobs}
        with patch.object(job_queue, 'list_jobs', return_value=jobs), \
                patch.object(job_queue, 'result', side_effect=lambda owner, job: records[job]), \
                patch.object(ui, 'latex'), patch.object(ui, 'solution_plots'), patch.object(ui, 'balance_plots'):
            app = AppTest.from_file(str(PIC / 'app/Home.py'), default_timeout=60).run()
            app.switch_page('views/12_Discovery_compare.py').run()
        self.assertFalse(app.exception)
        limits = [m.value for m in app.metric if m.label.startswith('noise limit')]
        self.assertEqual(limits, ['not established'])
        frame = next(d.value for d in app.dataframe if 'unscored runs' in d.value.columns)
        self.assertEqual(frame['runs'].tolist(), [3, 2])
        self.assertEqual(frame['unscored runs'].tolist(), [0, 2])
        self.assertTrue(frame.loc[1, 'success rate'] != frame.loc[1, 'success rate'])  # NaN plot gap


class DiscoverTranslation(unittest.TestCase):
    def ode(self):
        t = np.linspace(0, 1, 21)
        return Problem('test_ode', 'ODE', 'ode', 'synthetic', (t,), {'v': np.sin(t)}, ('t',))

    def run_result(self, feature, problem=None, method='discover'):
        problem = problem or self.ode()
        raw = {'targets': ['u_tt'], 'features': [[feature]], 'coefficients': [[0.5]],
               'model': 'mul(u1,diff(u1,x1))', 'discover_reward': 0.75}
        with patch.object(methods, '_bridge', return_value={'result': raw, 'settings': {}, 'fit_seconds': 1}):
            return methods.run(problem, problem.data, method)

    def test_ode_product_and_powers_use_time_derivatives(self):
        for feature in ('u u_x', 'u_x^2', 'u^2 u_x', 'u u_xx', 'u_x u_xx'):
            with self.subTest(feature=feature):
                front, _, _, info = self.run_result(feature)
                self.assertIsNone(info['scoring_note'])
                self.assertIn('/dx0', front[0][0])
                self.assertNotIn('u_x', info['result']['features'][0][0])
                self.assertEqual(info['result']['original_features'], [[feature]])
                self.assertEqual(info['result']['model'], 'mul(u1,diff(u1,x1))')
                self.assertEqual(info['result']['coefficients'], [[0.5]])

    def test_pde_products_keep_spatial_derivatives(self):
        grids = tuple(np.meshgrid(np.linspace(0, 1, 11), np.linspace(0, 1, 11), indexing='ij'))
        problem = Problem('test_pde', 'PDE', 'pde_1d', 'synthetic', grids, {'u': grids[0] + grids[1]}, ('t', 'x'))
        front, _, _, info = self.run_result('u u_x', problem)
        self.assertIn('du/dx1', front[0][0])
        self.assertEqual(info['result']['features'], [['u u_x']])
        self.assertNotIn('original_features', info['result'])

    def test_unknown_axis_preserves_native_result_without_fake_scoring(self):
        front, objectives, seconds, info = self.run_result('u_y')
        self.assertIn('u_y', front[0][0])
        self.assertTrue(info['scoring_note'])
        self.assertEqual(info['result']['discover_reward'], 0.75)
        self.assertNotIn('success_selected', methods.score(front, objectives, [], info))

    def test_conversion_valueerror_does_not_hide_non_discover_bugs(self):
        with patch('epde_bench.pdecomp.to_epde_texts', side_effect=ValueError('invalid axis')):
            from test_native_pdecomp_methods import heat
            with self.assertRaisesRegex(ValueError, 'invalid axis'):
                self.run_result('u', heat(), 'deepmod')


if __name__ == '__main__':
    unittest.main()
