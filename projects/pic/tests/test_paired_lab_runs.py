"""End-to-end scientific regression of the public Run a search paired experiment.

Execute ORIGINAL EPDE and PDEComp's own PySINDy against the exact same
scientific arrays (not a browser-only imitation). Retain and compare records
using one session capability and input_key as the Streamlit UI does.
"""
from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from epde_bench.config import load_config
from support.discovery_lab import job_queue as queue
from support.discovery_lab import worker


class PairedRunTests(unittest.TestCase):
    def test_original_epde_and_pdecomp_pysindy_both_saved_in_one_comparison_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous_root, previous_lock = queue.ROOT, queue.MANAGER_LOCK
            queue.ROOT = Path(tmp)
            queue.MANAGER_LOCK = queue.ROOT / "manager.lock"
            try:
                owner = queue.new_owner()
                search = copy.deepcopy(load_config("ode", "default")["search"])
                search["evolution"]["population_size"] = 4
                search["evolution"]["training_epochs"] = 1
                key = "record|ode|0|0"
                ids = []
                with patch.object(queue, "ensure_manager", return_value=None):
                    for method, variant in (("epde", "default"), ("pdecomp", "pysindy_pdecomp_fixed")):
                        ids.append(queue.submit(owner, {
                            "method": method, "variant": variant,
                            "source": "record", "dataset": "ode",
                            "noise": 0.0, "seed": 0, "effort": "custom",
                            "overrides": {"search": copy.deepcopy(search)} if method == "epde" else {},
                            "display": {"label": "Forced oscillator", "input_key": key},
                        }))
                self.assertEqual(len(ids), 2)
                self.assertNotEqual(ids[0], ids[1])
                for job_id in ids:
                    self.assertEqual(worker.main(queue.job_path(owner, job_id)), 0)
                    manifest = queue.get_job(owner, job_id)
                    manifest["state"] = "finished"
                    queue._write(queue.job_path(owner, job_id) / "job.json", manifest)
                rows = queue.list_jobs(owner)
                self.assertEqual(len(rows), 2)
                self.assertEqual({row["input_key"] for row in rows}, {key})
                self.assertEqual({row["method"] for row in rows}, {"epde", "pdecomp"})
                results = [queue.result(owner, job_id) for job_id in ids]
                for record in results:
                    self.assertEqual(record["status"], "ok")
                    self.assertTrue(record["front"], "Method returned no discovered candidate")
                    self.assertEqual(record["dataset"], "ode")
                    self.assertEqual(record["noise"], 0.0)
                    self.assertEqual(record["seed"], 0)
                    self.assertIn("evaluation", record)
                # Compare must NOT leak data from other sessions.
                self.assertEqual(queue.list_jobs(queue.new_owner()), [])
                self.assertEqual(
                    {row["method"]: record["method"] for row, record in zip(
                        [queue.get_job(owner, job_id) for job_id in ids], results)},
                    {"epde":"epde", "pdecomp":"pdecomp"})
            finally:
                queue.ROOT, queue.MANAGER_LOCK = previous_root, previous_lock


if __name__ == "__main__":
    unittest.main()
