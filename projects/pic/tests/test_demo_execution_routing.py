"""Server-side science resource routing; no Streamlit or EPDE dependency."""
from __future__ import annotations

import ast
from pathlib import Path
import unittest

APP = Path(__file__).resolve().parents[1] / "app"
import sys
sys.path.insert(0, str(APP))
from support.discovery_lab.execution_policy import (
    classify_epde, require_hosted_epde,
)


class HostedResourcePolicyTests(unittest.TestCase):
    def test_small_epde_ode_is_allowed(self):
        self.assertTrue(classify_epde("ode", (320,), 1,
                      {"evolution":{"population_size":16,"training_epochs":5}})[0])
        self.assertTrue(classify_epde("nir1_heat", (50, 50), 1,
                      {"evolution":{"population_size":8,"training_epochs":3}})[0])

    def test_heavy_pde_never_loads_on_public_worker(self):
        for name in ("ns", "jhtdb", "ks"):
            with self.subTest(record=name):
                self.assertFalse(classify_epde(name, (20,), 1, {})[0])
                with self.assertRaisesRegex(ValueError, "Hosted EPDE refused"):
                    require_hosted_epde(name, (20,), 1, {})

    def test_no_parameter_bypass(self):
        invalid = [
            ("huge dataset", None, (30, 400), 1, 8, 3),
            ("budget too large", "ode", (320,), 1, 48, 25),
            ("product bound", "heat", (50, 100), 1, 24, 6),
            ("three-dimensional", "ns_custom", (30, 30, 30), 1, 2, 1),
            ("multi-variable", "three_body", (200,), 3, 4, 2),
        ]
        for label, name, shape, variables, pop, epochs in invalid:
            with self.subTest(label=label):
                permitted, _, _ = classify_epde(name, shape, variables, {
                    "evolution": {"population_size":pop, "training_epochs":epochs}
                })
                self.assertFalse(permitted)

    def test_original_native_worker_bypasses_host_only_gate(self):
        # The local worker must still be able to run large PDEs with original EPDE.
        worker = (APP / "support/discovery_lab/worker.py").read_text()
        local = (APP / "local_worker_service.py").read_text()
        ast.parse(worker); ast.parse(local)
        self.assertIn("def _run_record(req, *, hosted=False):", worker)
        self.assertIn("def _run_upload(req, folder, *, hosted=False):", worker)
        self.assertIn("_run_record(job, hosted=hosted)", worker)
        self.assertIn("_run_upload(job, folder, hosted=hosted)", worker)
        self.assertIn("worker._run_record(runreq)", local)
        self.assertIn("worker._run_upload(runreq, target)", local)
        self.assertIn("EPDE_LOCAL_FULL_RUN", worker)

    def test_browser_sindy_route_has_no_server_cpu_run(self):
        browser = (APP / "views/13_Browser_sindy.py").read_text()
        lab = (APP / "views/10_Discovery_lab.py").read_text()
        self.assertIn("st.iframe(build_browser_html(embedded=True)", browser)
        self.assertNotIn("queue.submit", browser)
        self.assertIn("if is_local_only(record_name) and not local_full:", lab)
        self.assertIn("if location == \"Demo server\" and \"epde\" in methods:", lab)


if __name__ == "__main__":
    unittest.main()
