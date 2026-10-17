"""Local-worker security/protocol and UI smoke tests.

This deliberately avoids executing long scientific searches; the normal
native scientific smoke/campaign tests should run separately.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
import threading
import unittest
from http.client import HTTPConnection

APP = Path(__file__).resolve().parents[1] / "app"
sys.path[:0] = [str(APP), str(APP.parent), str(APP.parent.parent.parent)]
from local_worker_service import Handler, ThreadingHTTPServer, VALID_ORIGINS


class LocalCpuTests(unittest.TestCase):
    def test_local_worker_and_bridge_have_valid_syntax(self):
        for path in (APP / "local_worker_service.py",
                     APP / "support/discovery_lab/local_browser.py",
                     APP / "support/discovery_lab/job_queue.py",
                     APP / "views/10_Discovery_lab.py"):
            with self.subTest(file=path.name):
                ast.parse(path.read_text(encoding="utf-8"))
        js = (APP / "support/discovery_lab/local_browser_assets/local_bridge.js").read_text(
            encoding="utf-8")
        for term in ("fetch(", "/health", "/run", "setTriggerValue",
                     "user_computer", "127.0.0.1:8765"):
            self.assertIn(term, js)

    def test_loopback_only_config_and_explicit_authentication(self):
        from local_worker_service import HOST, PORT
        self.assertEqual(HOST, "127.0.0.1")
        self.assertEqual(PORT, 8765)
        self.assertIn("https://epde-explorer-demo.streamlit.app", VALID_ORIGINS)

    def test_http_origin_allowlist_and_auth(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.auth_token = "safe-token"
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            connection = HTTPConnection("127.0.0.1", server.server_port, timeout=3)
            connection.request("OPTIONS", "/run", headers={
                "Origin": "https://untrusted.invalid",
                "Access-Control-Request-Private-Network": "true",
            })
            res = connection.getresponse()
            self.assertEqual(res.status, 403)
            res.read()

            connection.request("OPTIONS", "/run", headers={
                "Origin": "https://epde-explorer-demo.streamlit.app",
                "Access-Control-Request-Private-Network": "true",
            })
            res = connection.getresponse()
            self.assertEqual(res.status, 204)
            self.assertEqual(res.getheader("Access-Control-Allow-Private-Network"), "true")
            res.read()

            connection.request("GET", "/health", headers={
                "Origin": "https://epde-explorer-demo.streamlit.app",
                "X-EPDE-Token": "wrong",
            })
            res = connection.getresponse()
            self.assertEqual(res.status, 403)
            res.read()

            connection.request("GET", "/health", headers={
                "Origin": "https://epde-explorer-demo.streamlit.app",
                "X-EPDE-Token": "safe-token",
            })
            res = connection.getresponse()
            self.assertEqual(res.status, 200)
            health = json.loads(res.read())
            self.assertEqual(health["execution_location"], "user_computer")
            connection.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)

    def test_lab_unifies_methods_without_execution_location_controls(self):
        lab = (APP / "views/10_Discovery_lab.py").read_text(encoding="utf-8")
        self.assertIn('st.session_state.setdefault("lab_discovery_methods", ["epde", "pdecomp"])', lab)
        self.assertIn('"method": m', lab)
        self.assertIn('"input_key": key', lab)
        self.assertIn("queue.submit(owner, req)", lab)
        self.assertIn('st.segmented_control("Data source"', lab)
        self.assertIn('"Dataset group"', lab)
        self.assertIn('Filter the full registry', lab)
        self.assertIn('wizard_progress(wizard_stage)', lab)
        self.assertIn('lab_draft_config', lab)
        self.assertIn('Complete effective scientific configuration', lab)
        for removed in ("lab_execution_location", "SERVER_MODE",
                        "Prepare local runs", "Local worker access code"):
            self.assertNotIn(removed, lab)


if __name__ == "__main__":
    unittest.main()
