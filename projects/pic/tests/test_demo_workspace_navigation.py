"""Guard the public demo's minimal navigation and shared state.

Source-level smoke checks run without starting Streamlit or expensive EPDE jobs.
They do not replace browser tests or numerical regression campaigns.
"""
from __future__ import annotations

import ast
from pathlib import Path
import unittest

APP = Path(__file__).resolve().parents[1] / "app"


class DemoNavigationTests(unittest.TestCase):
    def test_sources_have_valid_python_syntax(self):
        for file in (
            APP / "Home.py",
            APP / "views/10_Discovery_lab.py",
            APP / "views/11_Discovery_results.py",
            APP / "views/12_Discovery_compare.py",
            APP / "support/discovery_lab/session.py",
        ):
            with self.subTest(path=str(file)):
                ast.parse(file.read_text(encoding="utf-8"), filename=str(file))

    def test_default_sidebar_is_three_pages(self):
        content = (APP / "Home.py").read_text(encoding="utf-8")
        public_pages = content.split("# Preserve the old exploratory interface")[0]
        for view in ("10_Discovery_lab.py", "11_Discovery_results.py", "12_Discovery_compare.py"):
            self.assertEqual(public_pages.count(view), 1)
        self.assertEqual(public_pages.count("default=True"), 1)
        self.assertIn('title="Lab"', public_pages)
        self.assertIn('title="Results"', public_pages)
        self.assertIn('title="Compare"', public_pages)

    def test_one_launcher_and_two_distinct_review_pages(self):
        lab = (APP / "views/10_Discovery_lab.py").read_text(encoding="utf-8")
        results = (APP / "views/11_Discovery_results.py").read_text(encoding="utf-8")
        compare = (APP / "views/12_Discovery_compare.py").read_text(encoding="utf-8")
        self.assertIn("queue.submit(owner, request)", lab)
        self.assertNotIn("st.tabs(", lab)
        self.assertNotIn("queue.submit(", results)
        self.assertNotIn("queue.submit(", compare)
        self.assertIn("ui.solution_plots", results)
        self.assertIn("ui.solution_plots", compare)

    def test_all_pages_share_same_capability(self):
        for name in ("10_Discovery_lab.py", "11_Discovery_results.py", "12_Discovery_compare.py"):
            value = (APP / "views" / name).read_text(encoding="utf-8")
            self.assertIn("current_owner()", value)
        session = (APP / "support/discovery_lab/session.py").read_text(encoding="utf-8")
        self.assertIn('st.query_params["session"]', session)
        self.assertIn('st.session_state["discovery_lab_owner"]', session)

    def test_old_pages_are_private_opt_in(self):
        content = (APP / "Home.py").read_text(encoding="utf-8")
        self.assertIn('EPDE_SHOW_RESEARCH_PAGES', content)
        self.assertIn('EPDE_PUBLIC_DEMO', content)


if __name__ == "__main__":
    unittest.main()
