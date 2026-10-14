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
            APP / "views/13_Browser_sindy.py",
            APP / "views/12_Discovery_compare.py",
            APP / "views/9_How_EPDE_works.py",
            APP / "support/sindy_guide.py",
            APP / "support/discovery_lab/session.py",
        ):
            with self.subTest(path=str(file)):
                ast.parse(file.read_text(encoding="utf-8"), filename=str(file))

    def test_default_sidebar_keeps_lab_and_restores_learning(self):
        content = (APP / "Home.py").read_text(encoding="utf-8")
        public_pages = content.split("# Preserve the old exploratory interface")[0]
        for view in ("10_Discovery_lab.py", "11_Discovery_results.py",
                     "12_Discovery_compare.py", "13_Browser_sindy.py"):
            self.assertEqual(public_pages.count(view), 1)
        self.assertEqual(public_pages.count("default=True"), 1)
        self.assertIn('title="Lab · EPDE"', public_pages)
        self.assertIn('title="Results · server"', public_pages)
        self.assertIn('title="Compare · server"', public_pages)
        self.assertEqual(public_pages.count("9_How_EPDE_works.py"), 1)
        self.assertIn('title="How it works · EPDE & SINDy"', public_pages)

    def test_one_launcher_and_two_distinct_review_pages(self):
        lab = (APP / "views/10_Discovery_lab.py").read_text(encoding="utf-8")
        results = (APP / "views/11_Discovery_results.py").read_text(encoding="utf-8")
        compare = (APP / "views/12_Discovery_compare.py").read_text(encoding="utf-8")
        self.assertIn("queue.submit(owner, req)", lab)
        self.assertNotIn("st.tabs(", lab)
        self.assertNotIn("queue.submit(", results)
        self.assertNotIn("queue.submit(", compare)
        self.assertIn("ui.solution_plots", results)
        self.assertIn("ui.solution_plots", compare)

    def test_original_epde_lesson_and_likec4_are_preserved(self):
        epde = (APP / 'views/9_How_EPDE_works.py').read_text(encoding='utf-8')
        for essential in ('STAGES = {', 'STAGE_VIEWS = {', 'MODEL_VIEWS = {',
                          'diagrams.likec4_view(view, height=380, sequence=sequence)',
                          'Explore the whole architecture', 'stage_interfaces(chosen)',
                          'Run short evolution', 'graph(chosen)'):
            with self.subTest(essential=essential):
                self.assertIn(essential, epde)
        self.assertIn('render_sindy_guide()', epde)
        self.assertTrue((APP / 'static/likec4-views.js').is_file())

    def test_sindy_guide_uses_likec4_and_real_baseline_terms(self):
        guide = (APP / 'support/sindy_guide.py').read_text(encoding='utf-8')
        for term in ('benchView', 'picView', 'diagrams.likec4_view',
                     'STLSQ', '20%', 'run_pysindy', 'st.graphviz_chart'):
            with self.subTest(term=term):
                self.assertIn(term, guide)
        self.assertNotIn('queue.submit(', guide)

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
