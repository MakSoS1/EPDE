"""Learning-rate-free options of the DeepXDE fitness solve.

The pins in ``TestTheShippedSolveIsPinned`` were captured from the code BEFORE
any lr-free option existed (Sep 28 2026), on CPU. They cover the two seams the
new options touch and that no earlier test reached bitwise:

* the gradient-weight probe plus Adam plus L-BFGS (``epochs > 0``);
* the gradient-weight probe plus L-BFGS alone (``epochs = 0``).

``TestAFixedCoefficientDeepXDESolve`` uses 'variance' weights and no L-BFGS,
so it never passes through either. Every new option is opt-in, so the pins
must survive all of them unchanged.
"""
import hashlib

import numpy as np
import pytest
import torch

from test_solver_path import (_fitted_oscillator, fitted,          # noqa: F401
                              lbfgs_on_cpu)


@pytest.fixture(scope='module', autouse=True)
def _restore_torch_default_device():
    before = torch.get_default_device()
    yield
    torch.set_default_device(before)


@pytest.fixture
def solve_on_cpu(lbfgs_on_cpu, monkeypatch):                     # noqa: F811
    """Keep the solve on the CPU: ``_solve_device`` would otherwise re-enter
    DeepXDE's CUDA default for the length of the solve."""
    import epde.integrate.deepxde_integration as dxi
    monkeypatch.setattr(dxi, 'DDE_SOLVE_DEVICE', None)
    yield dxi


def _solve(dxi, search, config, seed=0):
    """One adapter solve on the fixture's first trajectory; returns the
    full-grid prediction, the reported loss and the adapter."""
    import deepxde as dde
    import epde.globals as global_var
    system = _fitted_oscillator(search)
    samples = global_var.samples_manager
    key = samples.trajecatoryIDs[0]
    observed = np.asarray(samples.get(('u', (1.0,)))[key]).reshape(-1)
    adapter = dxi.DeepXDEAdapter(**config)
    dde.config.set_random_seed(seed)
    solutions, loss = adapter.solve(system, grids=samples.grids()[key],
                                    data=[observed], domain_key=key)
    return np.asarray(solutions[0]), float(loss), adapter


def _digest(solution, loss):
    h = hashlib.sha256(np.ascontiguousarray(solution, dtype=np.float64).tobytes())
    h.update(repr(loss).encode())
    return h.hexdigest()[:24]


PIN_CONFIGS = {
    'adam+lbfgs': {'net': [16, 16], 'num_domain': 64, 'num_test': 16, 'epochs': 20,
                   'lr': 1e-4, 'lbfgs_maxiter': 20, 'train_frac': 0.8,
                   'loss_weight_mode': 'gradient'},
    'lbfgs-only': {'net': [16, 16], 'num_domain': 64, 'num_test': 16, 'epochs': 0,
                   'lr': 1e-4, 'lbfgs_maxiter': 20, 'train_frac': 0.8,
                   'loss_weight_mode': 'gradient'},
}
#: Captured Sep 28 2026 from the pre-change adapter, CPU, seed 0.
PIN_DIGESTS = {
    'adam+lbfgs': '6f9cc57457ed66c62a35fddb',
    'lbfgs-only': '361e35ab2ff3ef152e604911',
}


@pytest.mark.usefixtures('solve_on_cpu')
class TestTheShippedSolveIsPinned:

    @pytest.mark.parametrize('name', sorted(PIN_CONFIGS))
    def test_the_prediction_and_loss_are_bitwise_unchanged(self, name, fitted,   # noqa: F811
                                                            solve_on_cpu):
        solution, loss, adapter = _solve(solve_on_cpu, fitted, PIN_CONFIGS[name])
        assert np.isfinite(solution).all() and np.isfinite(loss)
        digest = _digest(solution, loss)
        print(f'PIN {name} {digest} loss={loss!r} exit={adapter.last_lbfgs_exit}')
        assert PIN_DIGESTS[name] is not None, f'capture: {name} -> {digest}'
        assert digest == PIN_DIGESTS[name]

    def test_a_repeat_in_the_same_process_is_bitwise_equal(self, fitted,          # noqa: F811
                                                           solve_on_cpu):
        """What the pins rest on: on the CPU the solve is a pure function of
        the config and the seed, whatever ran before it in the process."""
        first = _solve(solve_on_cpu, fitted, PIN_CONFIGS['lbfgs-only'])
        second = _solve(solve_on_cpu, fitted, PIN_CONFIGS['lbfgs-only'])
        assert _digest(first[0], first[1]) == _digest(second[0], second[1])


# ============================================================ the lr-free options
LBFGS_ONLY = PIN_CONFIGS['lbfgs-only']


@pytest.mark.usefixtures('solve_on_cpu')
class TestTheNullChecks:
    """Options that must change NOTHING when set to their null value -- the
    evidence that each arm's seam is clean, checked bitwise."""

    def test_an_lbfgs_only_run_needs_no_learning_rate(self, fitted, solve_on_cpu):  # noqa: F811
        with_lr = _solve(solve_on_cpu, fitted, LBFGS_ONLY)
        without = _solve(solve_on_cpu, fitted, dict(LBFGS_ONLY, lr=None))
        assert _digest(with_lr[0], with_lr[1]) == _digest(without[0], without[1])

    def test_a_zero_iteration_prefit_changes_nothing(self, fitted, solve_on_cpu):   # noqa: F811
        base = _solve(solve_on_cpu, fitted, LBFGS_ONLY)
        null = _solve(solve_on_cpu, fitted, dict(LBFGS_ONLY, init='data_prefit',
                                                 prefit_maxiter=0))
        assert _digest(base[0], base[1]) == _digest(null[0], null[1])

    def test_the_pairing_digest_changes_nothing(self, fitted, solve_on_cpu):        # noqa: F811
        base = _solve(solve_on_cpu, fitted, LBFGS_ONLY)
        paired = _solve(solve_on_cpu, fitted, dict(LBFGS_ONLY, pairing_digest=True))
        assert _digest(base[0], base[1]) == _digest(paired[0], paired[1])
        assert paired[2].last_pairing['theta0'] and paired[2].last_pairing['loss_weights']


class TestTheConfigValidation:

    @pytest.mark.parametrize('cfg', [{'lr': 'fast'}, {'lr': -1.0}, {'lr': None, 'epochs': 5},
                                     {'init': 'magic'}, {'precision': 'float16'},
                                     {'pde_loss': 'l1'}, {'input_transform': 'log'}])
    def test_a_bad_value_raises_instead_of_scoring_nan(self, cfg):
        from epde.integrate.deepxde_integration import DeepXDEAdapter, DeepXDEConfigError
        with pytest.raises(DeepXDEConfigError):
            DeepXDEAdapter(**cfg)


@pytest.mark.usefixtures('solve_on_cpu')
class TestTheStartOptions:

    def test_the_last_layer_is_the_least_squares_fit(self, solve_on_cpu):
        import deepxde as dde
        from epde.integrate.heldout import time_split
        dxi = solve_on_cpu
        adapter = dxi.DeepXDEAdapter(init='lstsq_last_layer')
        dde.config.set_random_seed(0)
        net = dde.nn.FNN([1, 8, 8, 1], 'tanh', 'Glorot normal')
        t = np.linspace(0.0, 2.0, 40)
        u = np.sin(3 * t)
        split = time_split(t, 0.8)
        stats = adapter._apply_init(net, t[:, None], [u], split)
        with torch.no_grad():
            h = torch.as_tensor(t[split.train][:, None], dtype=torch.float32)
            for lin in net.linears[:-1]:
                h = torch.tanh(lin(h))
        A = np.hstack([h.double().numpy(), np.ones((int(split.train.sum()), 1))])
        want = np.linalg.lstsq(A, u[split.train], rcond=None)[0]
        got = np.concatenate([net.linears[-1].weight.detach().double().numpy().ravel(),
                              net.linears[-1].bias.detach().double().numpy()])
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)
        assert stats['rank'] == 9 and np.isfinite(stats['cond'])

    def test_float64_starts_from_exactly_the_float32_weights_and_restores(self, solve_on_cpu):
        import deepxde as dde
        dxi = solve_on_cpu
        dde.config.set_random_seed(3)
        f32 = dxi.DeepXDEAdapter()._build_net([2, 8, 1])
        adapter = dxi.DeepXDEAdapter(precision='float64')
        before = (dde.config.real.precision, torch.get_default_dtype())
        dde.config.set_random_seed(3)
        with adapter._precision_scope():
            f64 = adapter._build_net([2, 8, 1])
            assert dde.config.real.precision == 64
        assert (dde.config.real.precision, torch.get_default_dtype()) == before
        for a, b in zip(f32.parameters(), f64.parameters()):
            assert b.dtype == torch.float64
            assert torch.equal(a.detach().double(), b.detach())

    def test_a_float64_solve_leaves_float32_behind(self, fitted, solve_on_cpu):     # noqa: F811
        import deepxde as dde
        before = (dde.config.real.precision, torch.get_default_dtype())
        _, loss, _ = _solve(solve_on_cpu, fitted, dict(LBFGS_ONLY, precision='float64'))
        assert np.isfinite(loss)
        assert (dde.config.real.precision, torch.get_default_dtype()) == before


class TestTheCancellationRatio:

    def test_it_is_bounded_and_has_a_finite_gradient_where_every_term_vanishes(self):
        from epde.integrate.residual_terms import cancellation_ratio
        a = torch.tensor([0.0, 1.0, -2.0, 0.5], requires_grad=True)
        b = torch.tensor([0.0, 1.0, 3.0, -0.5], requires_grad=True)
        r = a - b                                      # an equation a = b
        mass = a.abs() + b.abs()
        rho = cancellation_ratio(r, mass, torch)
        assert rho[0] == 0.0                           # 0/0: the equation holds
        assert bool((rho.abs() <= 1.0).all())
        (rho ** 2).sum().backward()
        assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()

    def test_the_inverse_mass_ignores_points_with_no_information(self):
        from epde.integrate.residual_terms import inverse_mass, mass_diagnostics
        np.testing.assert_array_equal(inverse_mass(np.array([0.0, 2.0])), [0.0, 0.5])
        d = mass_diagnostics(np.array([0.0, 1.0, 4.0]))
        assert d['zero_share'] == pytest.approx(1 / 3)

    @pytest.mark.parametrize('loss', ['sinv_live', 'sinv_data'])
    def test_a_scale_invariant_solve_is_finite(self, loss, fitted, solve_on_cpu):  # noqa: F811
        solution, value, adapter = _solve(solve_on_cpu, fitted, dict(LBFGS_ONLY, pde_loss=loss))
        assert np.isfinite(solution).all() and np.isfinite(value)
        assert adapter.last_sinv_stats['floors'] == [0.0]


@pytest.mark.usefixtures('solve_on_cpu')
class TestTheSharedDataFit:
    """One observation-only fit per dataset, reused as every candidate's start."""

    @pytest.fixture(autouse=True)
    def _empty_cache(self, solve_on_cpu):
        solve_on_cpu.DeepXDEAdapter._SHARED_FITS.clear()
        yield
        solve_on_cpu.DeepXDEAdapter._SHARED_FITS.clear()

    CFG = dict(LBFGS_ONLY, init='shared_data_fit', shared_fit_maxiter=50)

    def _setup(self, fitted):
        import epde.globals as global_var
        system = _fitted_oscillator(fitted)
        samples = global_var.samples_manager
        key = samples.trajecatoryIDs[0]
        observed = np.asarray(samples.get(('u', (1.0,)))[key]).reshape(-1)
        return system, samples, key, observed

    def test_the_second_solve_loads_the_first_ones_fit(self, fitted, solve_on_cpu):  # noqa: F811
        first = _solve(solve_on_cpu, fitted, self.CFG)
        second = _solve(solve_on_cpu, fitted, self.CFG)
        assert first[2].last_init_stats['source'] == 'fitted'
        assert second[2].last_init_stats['source'] == 'memory'
        assert _digest(first[0], first[1]) == _digest(second[0], second[1])

    def test_a_fit_loaded_from_disk_is_bitwise_the_fitted_one(self, fitted, solve_on_cpu,  # noqa: F811
                                                              tmp_path):
        cfg = dict(self.CFG, shared_fit_dir=str(tmp_path))
        fitted_run = _solve(solve_on_cpu, fitted, cfg)
        solve_on_cpu.DeepXDEAdapter._SHARED_FITS.clear()
        loaded_run = _solve(solve_on_cpu, fitted, cfg)
        assert loaded_run[2].last_init_stats['source'] == 'disk'
        assert _digest(fitted_run[0], fitted_run[1]) == _digest(loaded_run[0], loaded_run[1])

    def test_the_fit_never_reads_the_tail(self, fitted, solve_on_cpu):          # noqa: F811
        import deepxde as dde
        from epde.integrate.heldout import time_split
        dxi = solve_on_cpu
        system, samples, key, observed = self._setup(fitted)
        mask = np.asarray(samples.gFunc('m')[key]).reshape(-1)
        t_inner = np.asarray(samples.grids()[key][0]).reshape(-1)[mask]
        blank = observed.copy()
        blank[~time_split(t_inner, 0.8).train] = np.nan
        keys = []
        for data in (observed, blank):
            dxi.DeepXDEAdapter._SHARED_FITS.clear()
            adapter = dxi.DeepXDEAdapter(**self.CFG)
            dde.config.set_random_seed(0)
            adapter.prepare_shared_fit(system, samples.grids()[key], [data], key)
            keys.append(adapter.last_init_stats['key'])
        assert keys[0] == keys[1]

    def test_a_fixed_seed_start_is_shared_and_leaves_the_global_rng_alone(self, fitted,  # noqa: F811
                                                                          solve_on_cpu):
        import deepxde as dde
        dxi = solve_on_cpu
        system, samples, key, observed = self._setup(fitted)
        adapter = dxi.DeepXDEAdapter(**dict(self.CFG, shared_fit_seed=7))
        dde.config.set_random_seed(0)
        adapter.prepare_shared_fit(system, samples.grids()[key], [observed], key)
        k0 = adapter.last_init_stats['key']
        dde.config.set_random_seed(123)               # another candidate's own init
        adapter.prepare_shared_fit(system, samples.grids()[key], [observed], key)
        assert adapter.last_init_stats['key'] == k0   # same start whatever the global RNG
        net = dde.nn.FNN([1, 16, 16, 16, 16, 1], 'tanh', 'Glorot normal')
        state = torch.get_rng_state()
        adapter._shared_data_fit(net, torch.tensor([[0.0], [0.5], [1.0]]),
                                 np.array([[0.0], [1.0], [4.0]]))
        assert torch.equal(state, torch.get_rng_state())


# ============================================================ Stage 1b/1c mechanics
class TestTheLrFreeOptimisers:

    def test_dog_takes_the_distance_over_gradients_step(self):
        from epde.integrate.lr_free import DoG
        x = torch.tensor([3.0, -4.0], requires_grad=True)
        target = torch.tensor([1.0, 1.0])
        opt = DoG([x], reps_rel=1e-6)

        def closure():
            opt.zero_grad()
            loss = 0.5 * ((x - target) ** 2).sum()
            loss.backward()
            return loss

        x0 = x.detach().clone()
        g = (x0 - target)
        opt.step(closure)
        eta = 1e-6 * (1.0 + float(x0.norm())) / float(g.norm())   # rbar / sqrt(G)
        torch.testing.assert_close(x.detach(), x0 - eta * g)
        for _ in range(300):
            opt.step(closure)
        assert float(((x - target) ** 2).sum()) < 1e-6 * float((g ** 2).sum())

    def test_prodigy_grows_its_step_from_d0_and_converges(self):
        from epde.integrate.lr_free import Prodigy
        x = torch.tensor([3.0, -4.0], requires_grad=True)
        target = torch.tensor([1.0, 1.0])
        opt = Prodigy([x], d0=1e-6)

        def closure():
            opt.zero_grad()
            loss = 0.5 * ((x - target) ** 2).sum()
            loss.backward()
            return loss

        start = float(closure())
        for _ in range(500):
            opt.step(closure)
        assert opt.param_groups[0]['d'] > 1e-3            # grew six orders from d0
        assert float(closure()) < 1e-3 * start

    @pytest.mark.parametrize('cls, kw', [('DoG', {'reps_rel': 0.0}), ('Prodigy', {'d0': -1.0})])
    def test_a_non_positive_constant_is_refused(self, cls, kw):
        from epde.integrate import lr_free
        with pytest.raises(ValueError):
            getattr(lr_free, cls)([torch.zeros(2, requires_grad=True)], **kw)


class TestTheStage1bConfig:

    @pytest.mark.parametrize('cfg', [
        {'lr': None, 'epochs': 5},                                    # adam needs an lr
        {'lr_rule': 'prefit_distance', 'first_order': 'dog', 'lr': None},
        {'lr_rule': 'data_selected', 'epochs': 0, 'lr': None},
        {'second_order': 'lm', 'lbfgs_maxiter': 10},                  # LM replaces L-BFGS
        {'first_order': 'sgd'}, {'second_order': 'newton'}, {'lr_rule': 'guess'}])
    def test_an_incoherent_config_raises(self, cfg):
        from epde.integrate.deepxde_integration import DeepXDEAdapter, DeepXDEConfigError
        with pytest.raises(DeepXDEConfigError):
            DeepXDEAdapter(**cfg)

    @pytest.mark.parametrize('cfg', [
        {'first_order': 'dog', 'lr': None}, {'first_order': 'prodigy', 'lr': None},
        {'lr_rule': 'prefit_distance', 'lr': None}, {'lr_rule': 'data_selected', 'lr': None},
        {'second_order': 'lm', 'lbfgs_maxiter': 0, 'epochs': 0, 'lr': None}])
    def test_the_lr_free_configs_are_accepted(self, cfg):
        from epde.integrate.deepxde_integration import DeepXDEAdapter
        DeepXDEAdapter(**cfg)


SMALL_ADAM = {'net': [16, 16], 'num_domain': 64, 'num_test': 16, 'epochs': 20,
              'lbfgs_maxiter': 20, 'train_frac': 0.8, 'loss_weight_mode': 'gradient'}


@pytest.mark.usefixtures('solve_on_cpu')
class TestTheStage1bSolves:

    @pytest.mark.parametrize('extra', [{'first_order': 'dog', 'lr': None},
                                       {'first_order': 'prodigy', 'lr': None}])
    def test_an_lr_free_first_order_phase_solves(self, extra, fitted, solve_on_cpu):  # noqa: F811
        solution, loss, adapter = _solve(solve_on_cpu, fitted, dict(SMALL_ADAM, **extra))
        assert np.isfinite(solution).all() and np.isfinite(loss)

    @pytest.mark.parametrize('rule', ['prefit_distance', 'data_selected'])
    def test_a_data_driven_lr_is_finite_and_leaves_the_rng_alone(self, rule, fitted,  # noqa: F811
                                                                 solve_on_cpu):
        cfg = dict(SMALL_ADAM, lr_rule=rule, lr=None, prefit_maxiter=20)
        solution, loss, adapter = _solve(solve_on_cpu, fitted, cfg)
        stats = adapter.last_lr_stats
        assert stats['rule'] == rule and stats['lr'] > 0 and np.isfinite(loss)
        # the rule runs on copies inside fork_rng: calling it again draws nothing
        import deepxde as dde
        from epde.integrate.heldout import time_split
        dde.config.set_random_seed(0)
        net = dde.nn.FNN([1, 8, 1], 'tanh', 'Glorot normal')
        t = np.linspace(0.0, 1.0, 20)
        before = torch.get_rng_state()
        adapter._data_driven_lr(net, t[:, None], [np.sin(t)], time_split(t, 0.8))
        assert torch.equal(before, torch.get_rng_state())

    def test_levenberg_marquardt_reports_the_loss_deepxde_reports(self, fitted,      # noqa: F811
                                                                  solve_on_cpu):
        cfg = {'net': [8, 8], 'num_domain': 64, 'num_test': 16, 'epochs': 0,
               'lbfgs_maxiter': 0, 'second_order': 'lm', 'lm_maxiter': 15, 'lr': None,
               'train_frac': 0.8, 'loss_weight_mode': 'gradient'}
        solution, loss, adapter = _solve(solve_on_cpu, fitted, cfg)
        stats = adapter.last_lm_stats
        assert np.isfinite(solution).all()
        assert stats['accepted'] >= 1
        assert stats['history'][-1][1] < stats['history'][0][1]        # LM lowered the loss
        # ||R||^2 is DeepXDE's weighted loss at the same parameters
        assert stats['loss'] == pytest.approx(loss, rel=1e-4)
        assert adapter.last_lbfgs_exit['reason'].startswith('lm_')


#: Captured Sep 29 2026, before the Stage-2 transform refactor (CPU, seed 0).
S1_PIN = 'c28c0dc36d3b02622e576293'


@pytest.mark.usefixtures('solve_on_cpu')
class TestTheS1PathIsPinned:
    """The affine-input (S1) solve, bitwise, so the Stage-2 transform
    composition (periodic embedding) can be shown not to move it."""

    def test_the_affine_input_solve_is_unchanged(self, fitted, solve_on_cpu):      # noqa: F811
        solution, loss, _ = _solve(solve_on_cpu, fitted,
                                   dict(LBFGS_ONLY, input_transform='affine'))
        digest = _digest(solution, loss)
        print(f'S1PIN {digest}')
        assert S1_PIN is not None, f'capture: {digest}'
        assert digest == S1_PIN
