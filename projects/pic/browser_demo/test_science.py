"""Independent numerical checks of the browser engine in native CPython.

These tests cannot substitute for a browser/WASM integration test. CI runs both.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import science


class BrowserScienceTests(unittest.TestCase):
    def run_case(self, dataset, method, **changes):
        request = {"dataset":dataset, "method":method, "samples":401,
                   "noise":0, "seed":7, "degree":2, "threshold":0.05,
                   "derivatives":"finite", "population":28, "epochs":12}
        request.update(changes)
        return json.loads(science.run_job(json.dumps(request)))

    def test_canonical_ode_presets_have_finite_data(self):
        for name, schema in science.CASES.items():
            with self.subTest(name=name):
                t, X, names, truth, _ = science.load_data({
                    "dataset": name, "seed": 1, "samples":301,
                })
                self.assertEqual(len(t), 301)
                self.assertEqual(X.shape, (301, len(names)))
                self.assertEqual(len(truth), len(names))
                self.assertTrue((t[1:] > t[:-1]).all())

    def test_evolutionary_educational_baseline_recovers_decay(self):
        record = self.run_case("decay","evo_demo",degree=1)
        self.assertEqual(record["status"],"ok")
        self.assertGreater(record["validation"]["r2"], 0.95)
        self.assertIn("NOT EPDE", record["details"]["implementation"])
        self.assertEqual(record["method"], "evo_demo")

    def test_experimental_method_is_not_mislabeled_epde(self):
        with self.assertRaises(ValueError):
            self.run_case("decay","epde")

    def test_savgol_derivatives_are_finite(self):
        record = self.run_case("decay","evo_demo",derivatives="savgol",degree=1)
        self.assertGreater(record["validation"]["r2"], 0.9)

    def test_csv_upload_and_bad_inputs(self):
        text = "time,x\n" + "\n".join(
            "{:.3f},{:.6f}".format(i/100, 2.0 - .5*i/100) for i in range(150)
        )
        t, X, names, _, _ = science.load_data({
            "dataset":"csv", "csv":text, "seed":1, "noise":0
        })
        self.assertEqual(len(t),150)
        self.assertEqual(names, ["x"])
        self.assertEqual(X.shape,(150,1))
        for bad in ("time,x\n1,2\n0,3", "time,x\n0,nan", "time,u,u\n" +
                    "0,1,1\n"*90):
            with self.subTest(bad=bad[:15]), self.assertRaises(ValueError):
                science.load_data({"dataset":"csv","csv":bad})

    def test_params_resource_caps(self):
        with self.assertRaises(ValueError):
            self.run_case("decay","evo_demo",epochs=1000)
        with self.assertRaises(ValueError):
            self.run_case("decay","evo_demo",degree=12)

    def test_official_pysindy_regression_on_decay(self):
        try:
            import pysindy
        except ImportError:
            self.skipTest("Install official PySINDy to enable this native smoke test")
        record = self.run_case("decay","pysindy",degree=1)
        self.assertEqual(record["status"],"ok")
        self.assertEqual(record["details"]["implementation"],
                         "PySINDy 2.1.0 / STLSQ")
        self.assertGreater(record["validation"]["r2"],0.98)
        self.assertTrue(any("u" in eq for eq in record["equations"]))

    def test_official_pysindy_logistic(self):
        try:
            import pysindy
        except ImportError:
            self.skipTest("PySINDy not installed in local interpreter")
        record = self.run_case("logistic","pysindy",degree=2)
        self.assertEqual(record["method"],"pysindy")
        self.assertGreater(record["validation"]["r2"],0.9)


if __name__ == "__main__":
    unittest.main()
