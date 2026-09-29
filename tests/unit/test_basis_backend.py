"""The fixed-basis solver backend (``epde.integrate.basis_integration``).

No network and no learning rate: a Chebyshev/Fourier expansion whose
coefficients the same torch L-BFGS as DeepXDE's optimises. These pins cover
exactness where the answer is known, invariance to the held-out tail,
bit-reproducibility, and the property the backend exists for: it never
imports deepxde.
"""
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch

import epde
import epde.globals as global_var
from epde.integrate.basis_integration import (BasisAdapter, Chebyshev, Fourier,
                                              default_basis_config)
from epde.integrate.heldout import DeepXDEConfigError, time_split

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


@pytest.fixture(scope='module', autouse=True)
def _restore_torch_default_device():
    before = torch.get_default_device()
    yield
    torch.set_default_device(before)


def _search_1d(t, u, boundary=10):
    search = epde.EpdeSearch(use_solver=False, verbose_params={'show_iter_idx': False},
                             device='cpu')
    _, domain = search.createDomain(t, boundary_width=boundary, ID=0)
    search.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
    _, trajectory = search.createTrajectory({'u': u}, domain, cache_id=0)
    search.create_pool(data=[trajectory], max_deriv_order=(2,), data_fun_pow=3,
                       additional_tokens=[])
    return search


def _search_2d(t, x, u, boundary=(3, 4)):
    search = epde.EpdeSearch(use_solver=False, verbose_params={'show_iter_idx': False},
                             device='cpu')
    grids = np.meshgrid(t, x, indexing='ij')
    _, domain = search.createDomain((grids[0], grids[1]), boundary_width=boundary, ID=0)
    search.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
    _, trajectory = search.createTrajectory({'u': u}, domain, cache_id=0)
    search.create_pool(data=[trajectory], max_deriv_order=(1, 2), additional_tokens=[])
    return search


def _exact(search, text, weights):
    """Translate ``text`` and set its coefficients EXACTLY (no FD refit), so
    the equation's solution is the data."""
    from epde.interface.equation_translator import translate_equation
    from epde.operators.common.coeff_calculation import LinRegBasedCoeffsEquation
    system = translate_equation(text, search.pool, all_vars=['u'])
    eq = system.vals['u']
    eq.main_var_to_explain = 'u'
    system.use_default_singleobjective_function()
    eq.weights_internal = np.ones(len(eq.structure))
    eq.weights_internal_evald = True
    LinRegBasedCoeffsEquation().apply(eq, {})
    eq.weights_final = np.asarray(weights, dtype=float)
    return system


def _solve(system, config):
    samples = global_var.samples_manager
    key = samples.trajecatoryIDs[0]
    observed = np.asarray(samples.get(('u', (1.0,)))[key]).reshape(-1)
    adapter = BasisAdapter(**config)
    solutions, loss = adapter.solve(system, grids=samples.grids()[key],
                                    data=[observed], domain_key=key)
    return adapter, np.asarray(solutions[0]), loss, observed


def _held_out_rmse(adapter, solution, observed):
    samples = global_var.samples_manager
    key = samples.trajecatoryIDs[0]
    mask = np.asarray(samples.gFunc('m')[key]).reshape(-1)
    t_inner = np.asarray(samples.grids()[key][0]).reshape(-1)[mask]
    held = time_split(t_inner, adapter.train_frac).test
    pred = solution.reshape(-1)[mask][held]
    return float(np.sqrt(np.mean((pred - observed[held]) ** 2)))


# ------------------------------------------------------------ 1-D bases
class TestTheOneDimensionalBases:

    def test_chebyshev_derivatives_match_the_polynomial(self):
        basis = Chebyshev(0.0, 4.0, 8)
        x = np.linspace(0.0, 4.0, 11)
        coeffs = np.array([0.3, -1.0, 0.5, 0.25, 0.0, 0.1, -0.2, 0.05])
        from numpy.polynomial import chebyshev as C
        xm = 2 * x / 4.0 - 1.0
        for order in (0, 1, 2, 3):
            want = C.chebval(xm, C.chebder(coeffs, m=order) if order else coeffs) * (0.5 ** order)
            np.testing.assert_allclose(basis.matrix(x, order) @ coeffs, want,
                                       rtol=1e-12, atol=1e-12)

    def test_fourier_derivatives_are_analytic(self):
        basis = Fourier(-1.0, 2.0, 3)
        x = np.linspace(-1.0, 1.0, 9, endpoint=False)
        w = 2 * np.pi * 2 / 2.0
        c = np.zeros(basis.size)
        c[3] = 1.0                                     # cos(w_2 (x - lo))
        np.testing.assert_allclose(basis.matrix(x, 2) @ c, -w ** 2 * np.cos(w * (x + 1.0)),
                                   rtol=0, atol=1e-12)

    def test_doubling_keeps_the_coarse_coefficients_in_place(self):
        """Refinement warm-starts by zero-padding, so the coarse layout must be
        a prefix of the fine one."""
        x = np.linspace(-1.0, 1.0, 7)
        for coarse in (Chebyshev(-1.0, 1.0, 5), Fourier(-1.0, 2.0, 3)):
            fine = coarse.refined()
            np.testing.assert_array_equal(fine.matrix(x)[:, :coarse.size], coarse.matrix(x))


# ------------------------------------------------------------ config
class TestTheConfig:

    def test_an_unknown_key_is_refused(self):
        with pytest.raises(DeepXDEConfigError, match='unknown keys'):
            BasisAdapter(learning_rate=1e-3)

    @pytest.mark.parametrize('key,value', [('pde_loss', 'l1'), ('start', 'random'),
                                           ('refine', 'always'), ('precondition', 'adam')])
    def test_an_unknown_option_is_refused(self, key, value):
        with pytest.raises(DeepXDEConfigError):
            BasisAdapter(**{key: value})

    def test_there_is_no_learning_rate_anywhere(self):
        assert not any('lr' == k or 'learning' in k for k in default_basis_config())


# ------------------------------------------------------------ solves
class TestExactSolutions:

    def test_an_oscillator_in_the_span_is_recovered_in_the_tail(self):
        t = np.linspace(0.0, 4 * np.pi, 120)
        u = np.sin(t) + 2.0 * np.cos(t)
        search = _search_1d(t, u)
        system = _exact(search, '-1.0 * u{power: 1.0} + 0.0 = d^2u/dx0^2{power: 1.0}',
                        [-1.0, 0.0])
        adapter, sol, loss, obs = _solve(system, {'threads': 1})
        assert _held_out_rmse(adapter, sol, obs) < 1e-6 * np.std(u)
        assert adapter.last_solve_stats['resolved']

    def test_a_periodic_heat_equation_is_recovered_in_the_tail(self):
        D, k = 0.05, 2
        t = np.linspace(0.0, 1.0, 31)
        x = np.linspace(0.0, 2.0, 48, endpoint=False)
        T, X = np.meshgrid(t, x, indexing='ij')
        u = np.exp(-D * (k * np.pi) ** 2 * T) * np.sin(k * np.pi * X)
        search = _search_2d(t, x, u)
        system = _exact(search, f'{D} * d^2u/dx1^2{{power: 1.0}} + 0.0 = du/dx0{{power: 1.0}}',
                        [D, 0.0])
        adapter, sol, loss, obs = _solve(system, {'threads': 1})
        stats = adapter.last_solve_stats
        assert stats['periodic_axes'] == [1]
        assert _held_out_rmse(adapter, sol, obs) < 1e-4 * np.std(u)

    def test_a_nonlinear_equation_converges(self):
        """u' = u - u^3 from u(0) = 0.1: a logistic-like approach to 1."""
        t = np.linspace(0.0, 6.0, 150)
        u0 = 0.1
        u = u0 * np.exp(t) / np.sqrt(1.0 + u0 ** 2 * (np.exp(2 * t) - 1.0))
        search = _search_1d(t, u)
        system = _exact(search, '-1.0 * u{power: 3.0} + 1.0 * u{power: 1.0} + 0.0 = '
                                'du/dx0{power: 1.0}', [-1.0, 1.0, 0.0])
        adapter, sol, loss, obs = _solve(system, {'threads': 1})
        assert np.isfinite(loss)
        assert _held_out_rmse(adapter, sol, obs) < 1e-4


class TestTheTailIsNeverRead:

    def test_a_nan_tail_leaves_the_solution_bit_identical(self, monkeypatch):
        t = np.linspace(0.0, 4 * np.pi, 120)
        u = np.sin(t) + 2.0 * np.cos(t)
        search = _search_1d(t, u)
        system = _exact(search, '-1.0 * u{power: 1.0} + 0.0 = d^2u/dx0^2{power: 1.0}',
                        [-1.0, 0.0])
        cfg = {'threads': 1, 'lbfgs_maxiter': 300, 'refine': 'off'}
        _, clean, loss_clean, obs = _solve(system, cfg)

        samples = global_var.samples_manager
        key = samples.trajecatoryIDs[0]
        mask = np.asarray(samples.gFunc('m')[key]).reshape(-1)
        t_inner = np.asarray(samples.grids()[key][0]).reshape(-1)[mask]
        split = time_split(t_inner, 0.8)
        blank_obs = obs.copy()
        blank_obs[~split.train] = np.nan
        raw = BasisAdapter._raw_fields(key, ['u'])
        blank_raw = {k: v.astype(float).copy() for k, v in raw.items()}
        blank_raw['u'][t > split.t_train] = np.nan
        monkeypatch.setattr(BasisAdapter, '_raw_fields',
                            staticmethod(lambda key, names: blank_raw))
        adapter = BasisAdapter(**cfg)
        sols, loss = adapter.solve(system, grids=samples.grids()[key], data=[blank_obs],
                                   domain_key=key)
        np.testing.assert_array_equal(np.asarray(sols[0]), clean)
        assert loss == loss_clean


class TestReproducibility:

    def test_a_repeat_in_process_is_bit_identical(self):
        t = np.linspace(0.0, 4 * np.pi, 120)
        search = _search_1d(t, np.sin(t) + 2.0 * np.cos(t))
        system = _exact(search, '-1.0 * u{power: 1.0} + 0.0 = d^2u/dx0^2{power: 1.0}',
                        [-1.0, 0.0])
        cfg = {'threads': 1, 'lbfgs_maxiter': 200}
        a = _solve(system, cfg)
        b = _solve(system, cfg)
        np.testing.assert_array_equal(a[1], b[1])
        assert a[2] == b[2]

    def test_a_subprocess_solve_is_bit_identical_and_never_imports_deepxde(self):
        script = textwrap.dedent(f'''
            import sys, hashlib
            sys.path.insert(0, {REPO!r}); sys.path.insert(0, {os.path.dirname(__file__)!r})
            import numpy as np
            from test_basis_backend import _search_1d, _exact, _solve
            t = np.linspace(0.0, 4 * np.pi, 120)
            search = _search_1d(t, np.sin(t) + 2.0 * np.cos(t))
            system = _exact(search, '-1.0 * u{{power: 1.0}} + 0.0 = d^2u/dx0^2{{power: 1.0}}',
                            [-1.0, 0.0])
            _, sol, loss, _ = _solve(system, {{'threads': 1, 'lbfgs_maxiter': 200}})
            print('DIGEST', hashlib.sha256(sol.tobytes()).hexdigest(), repr(loss))
            print('DEEPXDE', 'deepxde' in sys.modules)
        ''')
        env = dict(os.environ, PYTHONIOENCODING='utf-8', OMP_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        runs = [subprocess.run([sys.executable, '-c', script], capture_output=True,
                               text=True, env=env, cwd=REPO, timeout=600)
                for _ in range(2)]
        lines = [[ln for ln in r.stdout.splitlines() if ln.startswith(('DIGEST', 'DEEPXDE'))]
                 for r in runs]
        assert all(len(x) == 2 for x in lines), [r.stderr[-2000:] for r in runs]
        assert lines[0][0] == lines[1][0]
        assert lines[0][1] == 'DEEPXDE False'


class TestThePredictionRefinementRule:

    def test_it_records_the_prediction_change_and_a_stop_reason(self):
        t = np.linspace(0.0, 1.0, 21)
        x = np.linspace(0.0, 2.0, 32, endpoint=False)
        T, X = np.meshgrid(t, x, indexing='ij')
        u = np.exp(-0.05 * (2 * np.pi) ** 2 * T) * np.sin(2 * np.pi * X)
        search = _search_2d(t, x, u)
        system = _exact(search, '0.05 * d^2u/dx1^2{power: 1.0} + 0.0 = du/dx0{power: 1.0}',
                        [0.05, 0.0])
        adapter, sol, loss, obs = _solve(system, {'threads': 1, 'refine': 'prediction',
                                                  'refine_tol': 1e-30, 'lbfgs_maxiter': 300})
        stats = adapter.last_solve_stats
        assert stats['refine_stop'] in ('converged', 'resolved', 'max_refinements')
        assert stats['levels'][0]['prediction_change'] is None
        if stats['n_levels'] > 1:
            assert stats['levels'][1]['prediction_change'] is not None
        assert all(np.isfinite(v) for v in stats['obs_rmse'])
