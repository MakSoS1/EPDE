"""Smoke tests of the Streamlit app: every page renders without an exception.
No evolutionary search is started. Skipped when Streamlit is not installed."""
import sys
import subprocess
import unittest
from pathlib import Path

PIC = Path(__file__).resolve().parents[1]
APP = PIC / 'app'
sys.path[:0] = [str(APP), str(PIC), str(PIC.parent.parent)]

try:
    from streamlit.testing.v1 import AppTest
except ImportError:                                   # pragma: no cover
    AppTest = None


@unittest.skipIf(AppTest is None, 'streamlit is not installed')
class PagesRender(unittest.TestCase):
    def run_page(self, path):
        app = AppTest.from_file(str(path), default_timeout=120)
        app.run()
        self.assertFalse(app.exception, [e.value for e in app.exception])
        return app

    def test_home(self):
        self.run_page(APP / 'Home.py')

    def test_home_from_repository_root(self):
        # A fresh interpreter must not inherit this module's PIC sys.path setup.
        code = (
            'import sys; from streamlit.testing.v1 import AppTest; '
            f'sys.path.insert(0, {str(APP)!r}); '
            f'app = AppTest.from_file({str(APP / "Home.py")!r}).run(timeout=30); '
            'assert not app.exception, [e.value for e in app.exception]'
        )
        result = subprocess.run([sys.executable, '-c', code], cwd=PIC.parent.parent,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_every_page(self):
        for path in sorted((APP / 'views').glob('*.py')):
            with self.subTest(page=path.name):
                self.run_page(path)

    def test_signal_check_runs(self):
        app = self.run_page(APP / 'views' / '2_Signal_check.py')
        app.button[0].click().run()
        self.assertFalse(app.exception, [e.value for e in app.exception])
        self.assertTrue(app.dataframe)


class EquationLatex(unittest.TestCase):
    def test_forced_oscillator(self):
        from support.equations import equation_latex
        text = ('-0.99 * du/dx0{power: 1.0} * sin{power: 1.0, freq: 2.0, dim: 0.0} + '
                '-4.0 * u{power: 1.0} + 1.5 * x{power: 1.0, dim: 0.0} + 0.0 = d^2u/dx0^2{power: 1.0}')
        self.assertEqual(equation_latex(text), r'u_{tt} = -0.99\,u_{t}\,\sin(2 t) - 4\,u + 1.5\,t')

    def test_system_and_space_axis(self):
        from support.equations import system_latex
        out = system_latex(['/ -1.0 * u{power: 1.0} * du/dx1{power: 1.0} = du/dx0{power: 1.0}',
                            '\\ 1.0 * u{power: 2.0} = dv/dx0{power: 1.0}'])
        self.assertIn(r'u_{t} = -u\,u_{x}', out)
        self.assertIn('{u}^{2}', out)


if __name__ == '__main__':
    unittest.main()
