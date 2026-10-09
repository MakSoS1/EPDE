import sys
import unittest
from pathlib import Path
import numpy as np
PIC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PIC), str(PIC / 'app'), str(PIC.parent.parent)]
from support.viewer import reconstruction_inputs, plot_pareto
from epde_bench import datasets

class ViewerTests(unittest.TestCase):
    def test_noise_realisation_used_in_display(self):
        problem = datasets.load('ode')
        record = {'dataset': 'ode', 'noise': 10, 'seed': 4, 'search_config': {'sample': 1}}
        data, search = reconstruction_inputs(record, problem)
        np.testing.assert_array_equal(data['u'], problem.noisy(10, 4)['u'])
        self.assertFalse(np.array_equal(data['u'], problem.data['u']))
        self.assertEqual(search, {'sample': 1})

    def test_baseline_record_gets_effective_preprocessor(self):
        from epde_bench import load_config
        problem = datasets.load('ode')
        record = {'dataset': 'ode', 'noise': 0, 'seed': 0,
                  'config': load_config('ode', 'pysindy'),
                  'baseline': {'differentiation': 'finite_difference'}}
        data, search = reconstruction_inputs(record, problem)
        self.assertEqual(search['preprocessing']['default_preprocessor_type'], 'FD')
        self.assertEqual(search['preprocessing']['preprocessor_kwargs'], {})

    def test_pareto_labels_keep_record_indices(self):
        import matplotlib.pyplot as plt
        fig = plot_pareto([None, [1, 2], [float('nan'), 0], [3, 4]], selected=3)
        self.assertEqual([text.get_text() for text in fig.axes[0].texts], ['1', '3'])
        np.testing.assert_array_equal(fig.axes[0].collections[1].get_offsets(), [[3, 4]])
        plt.close(fig)
