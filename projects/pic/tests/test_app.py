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
        latex = [e.value for e in app.latex]
        self.assertGreaterEqual(len(latex), 3)      # known law, then known and fitted law per noise level
        self.assertTrue(app.success or app.warning or app.error)

    def test_derivatives_compute(self):
        app = self.run_page(APP / 'views' / '3_Derivatives.py')
        app.button[0].click().run()
        self.assertFalse(app.exception, [e.value for e in app.exception])
        self.assertTrue(app.metric)
        self.assertTrue(app.info)


class TermComparison(unittest.TestCase):
    TRUTH = ('-4.0 * u{power: 1.0} + -1.0 * du/dx0{power: 1.0} * sin{power: 1.0, freq: 2.0, dim: 0.0} + '
             '1.5 * x{power: 1.0, dim: 0.0} = d^2u/dx0^2{power: 1.0}')

    def test_missing_extra_and_coefficients(self):
        from support.compare import term_rows
        found = ('-3.9 * u{power: 1.0} + 0.2 * u{power: 2.0} + 1.5 * x{power: 1.0, dim: 0.0} + 0.0 = '
                 'd^2u/dx0^2{power: 1.0}')
        rows = {r['term']: r for r in term_rows(found, self.TRUTH, ('t',))}
        self.assertEqual(rows['u_t·sin(2t)']['verdict'], 'missing')
        self.assertEqual(rows['u^2']['verdict'], 'extra term')
        self.assertAlmostEqual(rows['u']['found'], -3.9)
        self.assertIn('2.5%', rows['u']['verdict'])

    def test_other_target_is_rescaled(self):
        from support.compare import term_rows
        flipped = ('-0.25 * d^2u/dx0^2{power: 1.0} + -0.25 * du/dx0{power: 1.0} * sin{power: 1.0, freq: 2.0, dim: 0.0} + '
                   '0.375 * x{power: 1.0, dim: 0.0} = u{power: 1.0}')
        for row in term_rows(flipped, self.TRUTH, ('t',)):
            self.assertAlmostEqual(row['found'], row['known law'])

    def test_fitted_signal_law(self):
        from epde_bench.truthcheck import check
        from support.compare import fitted_equation
        _, report = check('ode', noise_levels=(0,), series=True)
        eq = report[0]['equations'][0]
        self.assertEqual(eq['target'].shape, eq['predicted'].shape)
        self.assertEqual(len(report[0]['times']), len(eq['target']))
        text = fitted_equation(eq)
        self.assertTrue(text.endswith('= d^2u/dx0^2{power: 1.0}'))
        self.assertIn('* u{power: 1.0}', text)


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
