#!/usr/bin/env python3
"""Build small historical execution metadata from one committed campaign.

No dataset arrays, current configurations, local result directories, equations,
or recovery scores are read by the application. Rebuild offline with this file.
"""
import argparse
import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path

BASE = 'projects/pic/experiments/baseline_clean_2026-10-09'
OUTPUT = 'projects/pic/app/support/discovery_lab/starting_evidence.json'
ENVIRONMENT_KEYS = ('platform', 'python', 'machine', 'cpu_count', 'blas_threads',
                    'numpy', 'scipy', 'torch', 'pysindy', 'cuda', 'epde_commit')
FORBIDDEN = {'equation', 'equations', 'truth', 'reference', 'references', 'metrics',
             'success_front', 'success_selected', 'hamming', 'hamming_min',
             'hamming_selected', 'coef', 'coef_error', 'best_index',
             'selected_index', 'selected', 'front'}
STATUS_COUNTS = {'ok': 'completed', 'timeout': 'timed_out', 'error': 'failed',
                 'failed': 'failed', 'unsupported': 'unsupported'}
LIMITATIONS = [
    'Execution completion does not establish equation recovery or scientific accuracy.',
    'Evidence applies only to the exact dataset, noise, configuration and recorded implementation.',
    'The campaign contains noise-free runs only; no evidence is transferred to another noise level.',
    'Historical elapsed times describe the recorded environment and are not runtime predictions.',
    'Completed-run medians exclude failed, unsupported and timed-out runs; timeouts are censored.',
    'Legacy PySINDy uses the historical EPDE-derived library, not the current PDEComp adapter.',
    'This campaign contains no DeepMoD or DISCOVER runs.',
]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def non_oracle(value):
    if isinstance(value, dict):
        return {k: non_oracle(v) for k, v in value.items() if k not in FORBIDDEN}
    if isinstance(value, list):
        return [non_oracle(v) for v in value]
    return value


def aggregate(records, campaign):
    """Group repeated seeds without comparing or selecting recovered models."""
    groups = {}
    for record in records:
        identity = record.get('identity') or {}
        config = record.get('config') or identity.get('config')
        if not isinstance(config, dict):
            raise ValueError('Historical run has no recorded configuration')
        dataset, variant = record['dataset'], record['variant']
        expected_method = {'default': 'epde', 'pysindy': 'pysindy'}.get(variant)
        method = record.get('method') or config.get('method')
        if not expected_method or method != expected_method:
            raise ValueError('Unsupported or inconsistent historical implementation')
        implementation = 'pysindy_legacy' if method == 'pysindy' else 'epde'
        noise = record['noise']
        if type(noise) not in (int, float) or not math.isfinite(noise) or noise < 0:
            raise ValueError('Invalid historical noise percentage')
        noise = float(noise)
        settings = non_oracle({k: config[k] for k in ('loader', 'search', 'pysindy')
                              if k in config})
        config_hash = digest(dict(settings, dataset=dataset, variant=variant, method=method))
        dependencies = identity.get('dependencies') or {}
        provenance = {}
        for name in ('data_sha256', 'source_sha256'):
            value = identity.get(name)
            if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('Missing or invalid historical provenance hash')
            provenance[name] = value
        key = (dataset, variant, method, noise, config_hash,
               provenance['data_sha256'], provenance['source_sha256'], digest(dependencies))
        if key not in groups:
            groups[key] = {
                'dataset': dataset, 'variant': variant, 'method': method,
                'implementation': implementation, 'noise_percent': noise,
                'settings_hash': digest(settings), 'config_sha256': config_hash,
                **provenance, 'tested_settings': settings, 'dependencies': dependencies,
                'seeds': [], 'completed': 0, 'timed_out': 0, 'failed': 0,
                'unsupported': 0, 'total': 0, '_fit': [], '_total': [], '_env': {},
            }
        entry = groups[key]
        seed = record['seed']
        if type(seed) is not int or seed in entry['seeds']:
            raise ValueError('Invalid or duplicate seed for an exact historical configuration')
        entry['seeds'].append(seed)
        status = record['status']
        if status not in STATUS_COUNTS:
            raise ValueError('Unknown historical execution status')
        entry[STATUS_COUNTS[status]] += 1
        entry['total'] += 1
        origin = 'run' if record.get('environment') else 'campaign'
        environment = record.get('environment') or campaign.get('environment') or {}
        environment = {k: environment[k] for k in ENVIRONMENT_KEYS if k in environment}
        environment_key = canonical([origin, environment])
        observed = entry['_env'].setdefault(environment_key, {
            'origin': origin, 'values': environment, 'runs': 0})
        observed['runs'] += 1
        if status == 'ok':
            for field, target in (('fit_seconds', '_fit'), ('total_seconds', '_total')):
                value = record.get(field)
                if type(value) in (int, float) and math.isfinite(value) and value >= 0:
                    entry[target].append(float(value))
    result = []
    for key in sorted(groups):
        entry = groups[key]
        entry['seeds'].sort()
        for source, name in (('_fit', 'fit'), ('_total', 'total')):
            values = entry.pop(source)
            entry[f'median_completed_{name}_seconds'] = round(statistics.median(values), 6) if values else None
            entry[f'completed_{name}_time_samples'] = len(values)
        environments = entry.pop('_env')
        entry['environments'] = [environments[k] for k in sorted(environments)]
        result.append(entry)
    return result


def build_index(root):
    """Read only committed blobs under the explicit campaign allowlist."""
    root = Path(root)
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True)
    paths = git('ls-tree', '-r', '-z', '--name-only', 'HEAD', '--', BASE + '/runs').split('\0')
    paths = sorted(p for p in paths if p.endswith('.json'))
    if not paths:
        raise ValueError('The allowed committed baseline campaign has no runs')
    input_manifest = []
    def read_committed_json(path):
        raw = subprocess.check_output(['git', '-C', str(root), 'show', 'HEAD:' + path])
        input_manifest.append({'path': path, 'sha256': hashlib.sha256(raw).hexdigest()})
        return json.loads(raw)
    records = []
    for path in paths:
        if not path.startswith(BASE + '/runs/') or len(Path(path).relative_to(BASE + '/runs').parts) != 2:
            raise ValueError('Unexpected path in historical campaign')
        record = read_committed_json(path)
        if Path(path).parent.name != record['dataset']:
            raise ValueError('Historical path and dataset disagree')
        records.append(record)
    campaign = read_committed_json(BASE + '/campaign.json')
    # Content provenance is stable in shallow checkouts and changes whenever
    # a committed input changes. Do not infer its historic commit from git log.
    input_hash = digest(sorted(input_manifest, key=lambda item: item['path']))
    return {'version': 1,
            'source': {'campaign': 'baseline_clean_2026-10-09', 'path': BASE,
                       'input_manifest_sha256': input_hash, 'run_count': len(records),
                       'timeout_seconds': campaign.get('timeout')},
            'limitations': LIMITATIONS, 'entries': aggregate(records, campaign)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    index = build_index(args.repo_root)
    output = args.output or args.repo_root / OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(index, ensure_ascii=True, separators=(',', ':')) + '\n')
    print(f'{len(index["entries"])} entries from {index["source"]["run_count"]} committed runs')


if __name__ == '__main__':
    main()
