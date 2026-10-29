"""Historical context must use committed metadata, without quality selection."""
import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PIC = Path(__file__).resolve().parents[1]
ROOT = PIC.parents[1]
SCRIPT = PIC / 'scripts/build_starting_evidence.py'
INDEX = PIC / 'app/support/discovery_lab/starting_evidence.json'
BASE = 'projects/pic/experiments/baseline_clean_2026-10-09'


def builder():
    spec = importlib.util.spec_from_file_location('build_starting_evidence', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record(seed=0, status='ok', method='epde', noise=0.0):
    variant = 'default' if method == 'epde' else 'pysindy'
    config = {'dataset': 'ode', 'variant': variant, 'method': method,
              'loader': {}, 'search': {'evolution': {'population_size': 16,
                                                     'training_epochs': 5}}}
    return {'dataset': 'ode', 'variant': variant, 'noise': noise, 'seed': seed,
            'method': method, 'status': status, 'config': config,
            'fit_seconds': 2 + seed, 'total_seconds': 3 + seed,
            'environment': {'python': '3.13.7', 'machine': 'arm64'},
            'identity': {'config': copy.deepcopy(config), 'data_sha256': '1' * 64,
                         'source_sha256': '2' * 64, 'dependencies': {'numpy': '2.3.4'}}}


class StartingEvidence(unittest.TestCase):
    def test_cli_rebuilds_the_published_index_from_allowed_committed_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'index.json'
            result = subprocess.run([sys.executable, str(SCRIPT), '--output', str(output)],
                                    cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            built = json.loads(output.read_text())
        self.assertEqual(built, json.loads(INDEX.read_text()))
        self.assertEqual(built['source']['run_count'], 150)
        self.assertEqual(len(built['entries']), 30)
        self.assertEqual(sum(e['completed'] for e in built['entries']), 130)
        self.assertEqual(sum(e['timed_out'] for e in built['entries']), 10)
        self.assertEqual(sum(e['unsupported'] for e in built['entries']), 10)
        self.assertTrue(all(e['total'] == 5 for e in built['entries']))

    def test_timeout_without_top_level_method_is_counted_and_not_timed_as_completed(self):
        module = builder()
        good = record()
        timeout = record(1, 'timeout')
        for key in ('method', 'config', 'environment', 'fit_seconds'):
            timeout.pop(key)
        timeout['total_seconds'] = 900
        failed = record(2, 'error')
        unsupported = record(3, 'unsupported')
        entry = module.aggregate([good, timeout, failed, unsupported],
                                 {'environment': {'python': '3.13.7'}})[0]
        self.assertEqual(entry['method'], 'epde')
        self.assertEqual([entry[k] for k in ('completed', 'timed_out', 'failed', 'unsupported', 'total')],
                         [1, 1, 1, 1, 4])
        self.assertEqual(entry['median_completed_fit_seconds'], 2)
        self.assertEqual(entry['median_completed_total_seconds'], 3)
        self.assertIn('campaign', [e['origin'] for e in entry['environments']])

    def test_exact_noise_configuration_and_provenance_are_not_merged(self):
        module = builder()
        original = record()
        noisy = record(noise=0.1)
        nearby_noise = record(noise=0.10001)
        changed_settings = record()
        changed_settings['config']['search']['evolution']['population_size'] = 24
        changed_settings['identity']['config'] = copy.deepcopy(changed_settings['config'])
        changed_source = record()
        changed_source['identity']['source_sha256'] = '3' * 64
        legacy = record(method='pysindy')
        entries = module.aggregate([original, noisy, nearby_noise, changed_settings,
                                    changed_source, legacy], {})
        self.assertEqual(len(entries), 6)
        old = next(e for e in entries if e['method'] == 'pysindy')
        self.assertEqual(old['implementation'], 'pysindy_legacy')
        self.assertFalse(any(e['method'] == 'pdecomp' for e in entries))

    def test_quality_equations_and_references_have_no_effect(self):
        module = builder()
        plain = record()
        scored = copy.deepcopy(plain)
        scored.update({'metrics': {'success_front': True, 'hamming_min': 0,
                                   'coef_error': 0, 'best_index': 3},
                       'front': ['secret recovered equation'],
                       'problem': {'truth': ['secret reference equation']},
                       'selected': {'equations': ['another equation']}})
        self.assertEqual(module.aggregate([plain], {}), module.aggregate([scored], {}))

    def test_builder_excludes_local_untracked_and_modified_results(self):
        module = builder()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign = root / BASE
            run = campaign / 'runs/ode/default__noise0__seed0.json'
            run.parent.mkdir(parents=True)
            run.write_text(json.dumps(record()))
            (campaign / 'campaign.json').write_text(json.dumps({'timeout': 900, 'environment': {}}))
            def git(*args):
                return subprocess.run(['git', '-C', str(root), *args], check=True,
                                      capture_output=True, text=True)
            git('init', '-q')
            git('add', '.')
            git('-c', 'user.name=Evidence test', '-c', 'user.email=test@example.invalid',
                '-c', 'commit.gpgsign=false', 'commit', '-qm', 'Baseline fixture')
            baseline = module.build_index(root)
            run.write_text(json.dumps(record(status='error')))
            (run.parent / 'default__noise0__seed1.json').write_text(json.dumps(record(1)))
            local = root / 'projects/pic/results'
            local.mkdir(parents=True)
            (local / 'extra.json').write_text(json.dumps(record(2)))
            (root / 'README.md').write_text('Unrelated application update\n')
            git('add', 'README.md')
            git('-c', 'user.name=Evidence test', '-c', 'user.email=test@example.invalid',
                '-c', 'commit.gpgsign=false', 'commit', '-qm', 'Unrelated update')
            index = module.build_index(root)
            self.assertEqual(index, baseline)
            shallow = root / 'shallow-checkout'
            subprocess.run(['git', 'clone', '--quiet', '--depth=1', root.as_uri(), str(shallow)],
                           check=True, capture_output=True, text=True)
            self.assertEqual(subprocess.check_output(
                ['git', '-C', str(shallow), 'rev-parse', '--is-shallow-repository'],
                text=True).strip(), 'true')
            self.assertEqual(module.build_index(shallow), index)
            git('add', str(run.relative_to(root)))
            git('-c', 'user.name=Evidence test', '-c', 'user.email=test@example.invalid',
                '-c', 'commit.gpgsign=false', 'commit', '-qm', 'Changed recorded input')
            changed = module.build_index(root)
            self.assertNotEqual(changed['source']['input_manifest_sha256'],
                                index['source']['input_manifest_sha256'])
            self.assertEqual(changed['entries'][0]['failed'], 1)
        self.assertNotIn('git_revision', index['source'])
        self.assertEqual(len(index['source']['input_manifest_sha256']), 64)
        self.assertEqual(index['source']['run_count'], 1)
        self.assertEqual(index['entries'][0]['completed'], 1)
        self.assertEqual(index['entries'][0]['failed'], 0)


if __name__ == '__main__':
    unittest.main()
