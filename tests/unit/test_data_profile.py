"""``epde.integrate.data_profile``: what the observed field says, from the
train window only."""
import math
import os

import numpy as np
import pytest

from epde.integrate.data_profile import (DataProfile, build_profile,
                                         gradient_density)

DATA = os.path.join(os.path.dirname(__file__), '..', '..', 'projects', 'pic', 'data')


def _grid(t, x):
    return np.meshgrid(np.asarray(t, float), np.asarray(x, float), indexing='ij')


def _t_train(n_levels, boundary, train_frac=0.8):
    """Last train level on the FULL grid, by ``time_split``'s rule on the
    inner domain (``n_fit = ceil(frac * n_inner)``)."""
    n_inner = n_levels - 2 * boundary
    return boundary + math.ceil(train_frac * n_inner) - 1


class TestPeriodicityVerdicts:

    def test_allen_cahn_is_periodic_in_x_with_period_two(self):
        path = os.path.join(DATA, 'ac', 'ac_data.npy')
        if not os.path.exists(path):
            pytest.skip('ac_data.npy not present')
        u = np.load(path)                                     # (51, 128)
        t = np.linspace(0.0, 1.0, 51)
        x = np.linspace(-1.0, 0.984375, 128)
        prof = build_profile(_grid(t, x), {'u': u}, t[_t_train(51, 5)])
        ax = prof.axes[1]
        assert ax.periodic and ax.endpoint == 'excluded'
        assert ax.period == pytest.approx(2.0, abs=1e-12)
        assert not prof.axes[0].periodic and prof.axes[0].seam_ratio is None

    def test_burgers_is_periodic_in_x_with_period_sixteen(self):
        path = os.path.join(DATA, 'burgers', 'burgers.mat')
        if not os.path.exists(path):
            pytest.skip('burgers.mat not present')
        from scipy.io import loadmat
        m = loadmat(path)
        t, x = np.ravel(m['t']), np.ravel(m['x'])
        u = np.real(m['usol']).T
        prof = build_profile(_grid(t, x), {'u': u}, t[_t_train(len(t), 20)])
        ax = prof.axes[1]
        assert ax.periodic and ax.endpoint == 'excluded'
        assert ax.period == pytest.approx(16.0, abs=1e-12)

    def test_a_dirichlet_standing_wave_is_not_periodic(self):
        """Wave data are value-periodic (zero at both ends) but not
        derivative-periodic; the second-difference test must catch it."""
        t = np.linspace(0.0, 1.0, 41)
        x = np.linspace(0.0, 1.0, 81)
        T, X = _grid(t, x)
        u = np.sin(np.pi * X) * np.cos(np.pi * T)
        prof = build_profile((T, X), {'u': u}, t[30])
        assert not prof.axes[1].periodic
        assert prof.axes[1].seam_ratio > 10.0

    @pytest.mark.parametrize('endpoint', ['included', 'excluded'])
    def test_one_full_period_is_periodic_under_its_own_convention(self, endpoint):
        t = np.linspace(0.0, 1.0, 21)
        x = np.linspace(0.0, 1.0, 64, endpoint=(endpoint == 'included'))
        T, X = _grid(t, x)
        u = np.sin(2 * np.pi * X) * np.exp(-T)
        prof = build_profile((T, X), {'u': u}, t[15])
        ax = prof.axes[1]
        assert ax.periodic and ax.endpoint == endpoint
        assert ax.period == pytest.approx(1.0, abs=1e-12)

    def test_half_a_period_is_not_periodic(self):
        t = np.linspace(0.0, 1.0, 21)
        x = np.linspace(0.0, 1.0, 64)
        T, X = _grid(t, x)
        prof = build_profile((T, X), {'u': np.sin(np.pi * X) + 0 * T}, t[15])
        assert not prof.axes[1].periodic

    def test_the_verdict_is_a_median_over_levels_not_a_unanimous_vote(self):
        """A non-periodic field at the first few levels (AC's initial
        condition) must not veto a field that is periodic afterwards."""
        t = np.linspace(0.0, 1.0, 21)
        x = np.linspace(0.0, 1.0, 64, endpoint=False)
        T, X = _grid(t, x)
        u = np.sin(2 * np.pi * X) + np.where(T < 0.2, X, 0.0)   # ramp early on
        prof = build_profile((T, X), {'u': u}, t[15])
        assert prof.axes[1].periodic


class TestSpectralSize:

    def test_a_three_mode_field_needs_exactly_its_highest_wavenumber(self):
        t = np.linspace(0.0, 1.0, 11)
        x = np.linspace(0.0, 1.0, 128, endpoint=False)
        T, X = _grid(t, x)
        u = (1.0 + np.sin(2 * np.pi * X) + 0.5 * np.cos(3 * 2 * np.pi * X)
             + 0.25 * np.sin(5 * 2 * np.pi * X)) * np.exp(-0.1 * T)
        prof = build_profile((T, X), {'u': u}, t[8])
        assert prof.axes[1].periodic
        assert prof.axes[1].modes == 5

    def test_time_counts_are_scaled_to_the_full_extent(self):
        """The basis must cover the tail too, so a train-window count is
        stretched by full / train extent."""
        t = np.linspace(0.0, 2.0, 201)
        u = np.cos(3.0 * t)[:, None] * np.ones((1, 8))
        x = np.linspace(0.0, 1.0, 8)
        T, X = _grid(t, x)
        half = build_profile((T, X), {'u': u}, t[100])
        full_like = build_profile((T, X), {'u': u}, t[-1])
        assert half.axes[0].modes >= full_like.axes[0].modes // 2
        assert half.axes[0].modes <= len(t)


class TestTrainWindowOnly:

    def test_a_nan_tail_leaves_the_profile_bit_identical(self):
        t = np.linspace(0.0, 1.0, 31)
        x = np.linspace(-1.0, 1.0, 64, endpoint=False)
        T, X = _grid(t, x)
        u = np.tanh(5 * np.cos(np.pi * X) * (1 + T))
        t_train = t[22]
        clean = build_profile((T, X), {'u': u}, t_train)
        blank = u.copy()
        blank[T > t_train] = np.nan
        assert build_profile((T, X), {'u': blank}, t_train) == clean
        d_clean = gradient_density((T, X), u, t_train)
        d_blank = gradient_density((T, X), blank, t_train)
        np.testing.assert_array_equal(d_clean, d_blank)

    def test_a_nan_inside_the_window_is_refused(self):
        t = np.linspace(0.0, 1.0, 11)
        x = np.linspace(0.0, 1.0, 16)
        T, X = _grid(t, x)
        u = np.sin(X) + T
        u[2, 3] = np.nan
        with pytest.raises(ValueError, match='not finite'):
            build_profile((T, X), {'u': u}, t[7])


class TestInvariances:

    @staticmethod
    def _field():
        t = np.linspace(0.0, 1.0, 26)
        x = np.linspace(-1.0, 1.0, 96, endpoint=False)
        T, X = _grid(t, x)
        u = np.tanh(3 * np.cos(np.pi * X) * (1 + T)) + 0.2 * np.sin(3 * np.pi * X)
        return t, x, T, X, u

    def test_affine_changes_of_the_field_change_only_its_moments(self):
        t, x, T, X, u = self._field()
        a = build_profile((T, X), {'u': u}, t[18])
        b = build_profile((T, X), {'u': 7.5 * u - 3.0}, t[18])
        for pa, pb in zip(a.axes, b.axes):
            assert (pa.periodic, pa.endpoint, pa.modes) == (pb.periodic, pb.endpoint, pb.modes)
            if pa.seam_ratio is not None:
                assert pb.seam_ratio == pytest.approx(pa.seam_ratio, rel=1e-9)
        va, vb = a.variable('u'), b.variable('u')
        assert vb.mean == pytest.approx(7.5 * va.mean - 3.0, rel=1e-12)
        assert vb.std == pytest.approx(7.5 * va.std, rel=1e-12)

    def test_rescaling_a_coordinate_rescales_only_its_period(self):
        t, x, T, X, u = self._field()
        a = build_profile((T, X), {'u': u}, t[18])
        b = build_profile((T, 4.0 * X + 1.0), {'u': u}, t[18])
        assert b.axes[1].periodic == a.axes[1].periodic
        assert b.axes[1].modes == a.axes[1].modes
        assert b.axes[1].period == pytest.approx(4.0 * a.axes[1].period, rel=1e-12)
        s, c = b.axes[1].affine
        np.testing.assert_allclose(s * (4.0 * x + 1.0) + c,
                                   a.axes[1].to_unit(x), rtol=0, atol=1e-12)

    def test_the_profile_is_deterministic(self):
        t, x, T, X, u = self._field()
        assert build_profile((T, X), {'u': u}, t[18]) == build_profile((T, X), {'u': u}, t[18])


class TestOrdinaryDifferentialEquations:

    def test_a_one_dimensional_series_has_only_a_time_axis(self):
        t = np.linspace(0.0, 20.0, 1001)
        u = np.cos(1.3 * t) + 0.1 * np.sin(4.1 * t)
        prof = build_profile([t], {'u': u}, t[800])
        assert isinstance(prof, DataProfile)
        assert len(prof.axes) == 1 and not prof.axes[0].periodic
        assert prof.periodic_axes == ()
        assert 1 <= prof.axes[0].modes <= len(t)
        assert prof.n_train_levels == 801


class TestGradientDensity:

    def test_a_constant_field_gives_a_flat_density(self):
        t = np.linspace(0.0, 1.0, 11)
        x = np.linspace(0.0, 1.0, 16)
        T, X = _grid(t, x)
        np.testing.assert_array_equal(gradient_density((T, X), 0 * T + 2.0, t[7]),
                                      np.ones_like(T))

    def test_the_density_is_highest_at_the_front_and_averages_two(self):
        t = np.linspace(0.0, 1.0, 11)
        x = np.linspace(-1.0, 1.0, 201)
        T, X = _grid(t, x)
        u = np.tanh(X / 0.05) + 0 * T
        d = gradient_density((T, X), u, t[7], kind='space')
        assert np.argmax(d[3]) == 100
        assert d[:8].mean() == pytest.approx(2.0, rel=1e-12)
