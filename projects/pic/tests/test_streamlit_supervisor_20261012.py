"""2026-10-12 scientific-supervisor Streamlit regression tests."""
import sys
import unittest
from pathlib import Path

PIC = Path(__file__).resolve().parents[1]
APP = PIC / "app"
sys.path[:0] = [str(APP), str(PIC), str(PIC.parent.parent)]


class ConfigImport(unittest.TestCase):
    def test_roundtrip(self):
        from support.config_io import parse_search_yaml, dump_search_yaml
        value = {"search": {"preprocessing": {"default_preprocessor_type": "poly",
                                              "preprocessor_kwargs": {
                                                  "poly_order": 5, "polynomial_window": 11}},
                            "search_space": {"tokens": [{
                                "family": "inverse_space_coordinate", "axis": 1}]}}}
        self.assertEqual(parse_search_yaml(dump_search_yaml(value).encode()), value)

    def test_no_unsafe_tags_or_arbitrary_functions(self):
        from support.config_io import parse_search_yaml
        for raw in (
            b"!!python/object/apply:os.system ['echo unsafe']",
            b"search: {search_space: {tokens: [{family: unsafe_python}]}}",
            b"seed: 1",
            b"search: {evolution: {population_size: 1000000}}",
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_search_yaml(raw)


class PolynomialSafety(unittest.TestCase):
    def test_reject_underdetermined_degree_and_windows(self):
        from epde_bench.poly_options import polynomial_kwargs
        with self.assertRaisesRegex(ValueError, "cannot compute derivative"):
            polynomial_kwargs(9, 3, [1, 4], (31, 31))
        with self.assertRaisesRegex(ValueError, "odd window"):
            polynomial_kwargs(6, 6, [1, 4], (31, 31))
        with self.assertRaisesRegex(ValueError, "grid points"):
            polynomial_kwargs(7, 6, [1, 4], (31, 31))
        with self.assertRaisesRegex(ValueError, "samples"):
            polynomial_kwargs(9, 5, [1, 4], (8, 31))
        self.assertEqual(polynomial_kwargs(11, 5, [1, 4], (31, 31))["poly_order"], 5)

    def test_fd_not_affected(self):
        from epde_bench.poly_options import validate_polynomial_options
        validate_polynomial_options({"preprocessing": {
            "default_preprocessor_type": "FD", "max_deriv_order": [6],
            "preprocessor_kwargs": {"polynomial_window": 2}}}, shape=(3,))


class PhysicalLawDiagnostic(unittest.TestCase):
    def test_intercept_opt_in_only(self):
        from epde_bench.truthcheck import check
        _, x = check("ode", noise_levels=(0.0,), series=True)
        eq = x[0]["equations"][0]
        self.assertFalse(eq["intercept_fitted"])
        self.assertEqual(eq["intercept"], 0.0)
        _, y = check("ode", noise_levels=(0.0,), series=True, include_intercept=True)
        eq2 = y[0]["equations"][0]
        self.assertTrue(eq2["intercept_fitted"])
        self.assertTrue(isinstance(eq2["intercept"], float))
        self.assertEqual(len(eq["terms"]), len(eq2["terms"]))


try:
    from streamlit.testing.v1 import AppTest
except ImportError:
    AppTest = None


@unittest.skipIf(AppTest is None, "Streamlit not installed")
class BrowserSmoke(unittest.TestCase):
    def test_new_sindy_screen_and_search(self):
        for p in ("4_SINDy.py", "5_Run_search.py",
                  "2_Signal_check.py", "3_Derivatives.py", "6_Your_data.py"):
            with self.subTest(p=p):
                app = AppTest.from_file(str(APP / "views" / p), default_timeout=120)
                app.run()
                self.assertFalse(app.exception, [e.message for e in app.exception])


if __name__ == "__main__":
    unittest.main()
