"""Published chronological evidence remains readable beside the search UI."""
import sys
import unittest
from pathlib import Path

PIC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PIC / 'app'), str(PIC), str(PIC.parent.parent)]

from streamlit.testing.v1 import AppTest


class HeldoutPage(unittest.TestCase):
    def test_published_case_and_runnable_interface_are_visible(self):
        app = AppTest.from_file(str(PIC / 'app/views/5_Run_search.py')).run(timeout=60)
        self.assertFalse(app.exception, [e.value for e in app.exception])
        self.assertTrue(any(b.label == 'Run training-only example' for b in app.button))
        self.assertTrue(any('run_heldout' in c.value for c in app.code))
        self.assertTrue(any('Published execution' in c.value for c in app.caption))
        self.assertTrue(app.latex)
