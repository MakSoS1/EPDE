import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
PIC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PIC / 'app'), str(PIC), str(PIC.parent.parent)]
from support import jobs

class Lifecycle(unittest.TestCase):
    def test_manifest_is_not_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'campaign.json').write_text(json.dumps({'jobs': [{'dataset': 'ode', 'variant': 'default', 'noise': 0, 'seed': 0}]}))
            job = {'kind': 'campaign', 'pid': 99999999, 'out': tmp}
            self.assertEqual(jobs.status(job), 'stopped')

    def test_supervisor_persists_exit_after_parent_reload(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(jobs, 'JOBS_DIR', Path(tmp)):
            job = jobs._start('run', 'fails', [sys.executable, '-c', 'import sys; sys.exit(7)'], Path(tmp) / 'result.json')
            for _ in range(150):
                stored = json.loads((Path(tmp) / (job['id'] + '.json')).read_text())
                if 'exit_code' in stored:
                    break
                time.sleep(.02)
            self.assertEqual(stored.get('exit_code'), 7)
            self.assertEqual(jobs.status(stored), 'error')

    def test_identity_mismatch_cannot_stop_an_unrelated_process(self):
        import os
        import psutil
        job = {'pid': os.getpid(), 'process_created': psutil.Process().create_time() - 10}
        with patch.object(psutil.Process, 'kill') as kill:
            jobs.stop(job)
            kill.assert_not_called()

    def test_stop_supervisor_persists_killed_child_exit(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(jobs, 'JOBS_DIR', Path(tmp)):
            job = jobs._start('run', 'sleeps', [sys.executable, '-c', 'import time; time.sleep(30)'], Path(tmp) / 'result.json')
            path = Path(tmp) / (job['id'] + '.json')
            for _ in range(150):
                stored = json.loads(path.read_text())
                if 'worker_pid' in stored:
                    break
                time.sleep(.02)
            self.assertTrue(jobs.stop(stored))
            for _ in range(150):
                stored = json.loads(path.read_text())
                if 'exit_code' in stored:
                    break
                time.sleep(.02)
            self.assertLess(stored['exit_code'], 0)
            self.assertEqual(jobs.status(stored), 'stopped')

    def test_progress_ignores_extra_invalid_and_wrong_identity_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = dict(dataset='ode', variant='default', noise=0, seed=0, identity='planned')
            (root / 'campaign.json').write_text(json.dumps({'jobs': [plan]}))
            run_dir = root / 'runs' / 'ode'
            run_dir.mkdir(parents=True)
            (run_dir / 'extra.json').write_text('{}')
            path = run_dir / 'default__noise0__seed0.json'
            path.write_text(json.dumps(dict(plan, identity='other', status='ok')))
            self.assertEqual(jobs.campaign_records(root), ([], 1))
            path.write_text(json.dumps(dict(plan, status='error')))
            job = dict(kind='campaign', out=str(root), pid=None, state='finished', exit_code=0)
            self.assertEqual(jobs.status(job), 'error')
            path.write_text(json.dumps(dict(plan, status='ok')))
            self.assertEqual(jobs.status(job), 'ok')

    def test_partial_campaign_is_distinct_from_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plans = [dict(dataset='ode', variant='default', noise=0, seed=i) for i in (0, 1)]
            (root / 'campaign.json').write_text(json.dumps({'jobs': plans}))
            run_dir = root / 'runs' / 'ode'
            run_dir.mkdir(parents=True)
            (run_dir / 'default__noise0__seed0.json').write_text('{"status": "ok"}')
            self.assertEqual(jobs.status(dict(kind='campaign', out=tmp, pid=None, state='finished', exit_code=0)), 'partial')

    def test_success_requires_a_valid_result_and_persists_zero_exit(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(jobs, 'JOBS_DIR', Path(tmp)):
            out = Path(tmp) / 'result.json'
            cmd = [sys.executable, '-c', 'import pathlib,sys; pathlib.Path(sys.argv[1]).write_text(\'{"status":"ok"}\')', str(out)]
            job = jobs._start('run', 'success', cmd, out)
            path = Path(tmp) / (job['id'] + '.json')
            for _ in range(150):
                stored = json.loads(path.read_text())
                if 'exit_code' in stored:
                    break
                time.sleep(.02)
            self.assertEqual(stored['exit_code'], 0)
            self.assertEqual(jobs.status(stored), 'ok')
            out.unlink()
            self.assertEqual(jobs.status(stored), 'error')

    def test_latest_attempt_is_sorted_by_start_time_not_random_id(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(jobs, 'JOBS_DIR', Path(tmp)):
            (Path(tmp) / '9999.json').write_text(json.dumps(dict(id='9999', kind='campaign', started=1)))
            (Path(tmp) / '0000.json').write_text(json.dumps(dict(id='0000', kind='campaign', started=2)))
            self.assertEqual([j['id'] for j in jobs.jobs('campaign')], ['0000', '9999'])

    def test_fresh_starting_job_is_running_before_supervisor_saves_pid(self):
        job = dict(kind='run', state='starting', started=time.time(), out='/nonexistent/result.json')
        self.assertEqual(jobs.status(job), 'running')

    def test_stale_starting_job_without_supervisor_pid_is_stopped(self):
        job = dict(kind='run', state='starting', started=time.time()-60, out='/nonexistent/result.json')
        self.assertEqual(jobs.status(job), 'stopped')
