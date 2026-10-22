"""Regressions found by reviewing actual desktop/mobile native-method runs."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PIC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PIC / 'app'), str(PIC), str(PIC.parents[1])]

from support.equations import equation_latex, native_equation_latex
from support.discovery_lab import ui


class NativeResultDisplay(unittest.TestCase):
    def test_long_equations_wrap_without_dropping_terms(self):
        expression = '1 * u{power: 1.0} + -2 * du/dx1{power: 1.0} + 1.2 = du/dx0{power: 1.0}'
        rendered = equation_latex(expression, wrap=True)
        self.assertIn(r'\begin{aligned}', rendered)
        self.assertEqual(rendered.count(r'\\'), 2)
        for term in ('u_{t}', '2\\,u_{x}', '1.2'):
            self.assertIn(term, rendered)

    def test_native_symbols_match_the_uploaded_variable_and_coordinates(self):
        rendered = native_equation_latex('0.5 * mul(sin(u1),cos(x1)) = u_t', ('time',), ('v',))
        self.assertIn(r'\sin\left(v\right)', rendered)
        self.assertIn(r'\cos\left(\mathrm{time}\right)', rendered)
        self.assertTrue(rendered.startswith('v_{t} = '))
        self.assertNotIn('u1', rendered)

    def test_native_operator_keeps_its_argument(self):
        rendered = native_equation_latex('0.1633 * diff2(u1,mul(cos(x1),sin(x1))) = u_t')
        self.assertIn(r'\operatorname{diff}_{2}', rendered)
        self.assertIn(r'\cos\left(x\right)\,\sin\left(x\right)', rendered)

    def test_native_rendering_preserves_algebraic_grouping(self):
        rendered = native_equation_latex('0.5 * (u + x1) * cos(x1) = u_t')
        self.assertIn(r'\left(u + x\right)', rendered)

    def test_unknown_expression_uses_display_fallback(self):
        with self.assertRaises(ValueError):
            native_equation_latex('some_unknown_operator(u1) = u_t')

    def test_large_native_expressions_have_a_wrapping_text_fallback(self):
        expression = '0.5 * ' + 'sin(' * 40 + 'u1' + ')' * 40 + ' = u_t'
        record = {'baseline': {'scoring_note': 'unavailable'}, 'problem': {'variables': ['u']}}
        with patch.object(ui.st, 'latex') as latex, patch.object(ui.st, 'code') as code:
            ui.latex([expression], ['t', 'x'], record=record)
        latex.assert_not_called()
        code.assert_called_once_with(expression, language=None, wrap_lines=True)

    def test_ode_plots_use_the_first_available_solution(self):
        missing = {'method': 'discover', 'evaluation': {'solution_note': 'unavailable'}}
        solved = {'method': 'pdecomp', 'evaluation': {'kind': 'ode', 'solution': {
            't': [0, 1, 2], 'observed': {'u': [1, 2, 3]}, 'simulated': {'u': [1, 2.1, 3.1]}}}}
        import matplotlib.pyplot as plt
        with patch.object(ui, 'show_figure', side_effect=plt.close) as show:
            ui.solution_plots([('DISCOVER', missing), ('PySINDy', solved)])
        self.assertEqual(show.call_count, 2)


class NativeTimeouts(unittest.TestCase):
    def test_native_runner_reports_timeout_without_a_traceback(self):
        from epde_bench.runner import run_one
        with patch('epde_bench.pdecomp_methods.run', side_effect=TimeoutError('Training time limit')):
            record = run_one('nir1_heat', 'deepmod_pdecomp_native')
        self.assertEqual(record['status'], 'timeout')
        self.assertEqual(record['error'], 'Training time limit')
        self.assertNotIn('traceback', record)


if __name__ == '__main__':
    unittest.main()
