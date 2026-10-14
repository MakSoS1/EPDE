import numpy as np
import pytest
from scipy.optimize import minimize

from projects.nir1_stability.nir1.design import ResearchDesign
from projects.nir1_stability.nir1.diagnostics import estimate_diagnostics
from projects.nir1_stability.nir1.fixtures import make_fixture
from projects.nir1_stability.nir1.regularizers import (
    build_penalty_weights, fit_weighted_en, post_refit,
)


def toy(X, y, names=None):
    return ResearchDesign(X, y, np.ones(len(y)), np.zeros(len(y)),
                          names or tuple(f"x{i}" for i in range(X.shape[1])), {})


def test_weighted_en_matches_closed_form_on_orthogonal_design():
    X = np.eye(3) * np.sqrt(3)
    d = toy(X, X @ np.array([2., -1., .1]))
    fit = fit_weighted_en(d, l1=.4, l2=.1, weights=np.ones(3))
    np.testing.assert_allclose(fit.beta, np.array([1.6 / 1.1, -.6 / 1.1, 0.]), atol=1e-7)
    assert fit.converged
    assert fit.kkt_residual < 1e-7


def test_correlated_design_objective_agrees_with_independent_scipy_reference():
    d, _ = make_fixture("T2", seed=0, correlation=.9)
    lam, l2 = .025, .01
    fit = fit_weighted_en(d, l1=lam, l2=l2, weights=np.ones(4))
    scales = np.sqrt(np.sum(d.X**2, axis=0) / len(d.y))
    A = d.X / scales
    def objective(z):
        coef, abs_slack = z[:4], z[4:]
        return .5 * np.mean((d.y - A @ coef)**2) + lam * np.sum(abs_slack) + .5 * l2 * np.sum(coef**2)
    constraints = [{"type": "ineq", "fun": lambda z: z[4:] - z[:4]},
                   {"type": "ineq", "fun": lambda z: z[4:] + z[:4]}]
    ref = minimize(objective, np.r_[np.zeros(4), np.ones(4)], method="SLSQP",
                   bounds=[(None, None)] * 4 + [(0, None)] * 4,
                   constraints=constraints, options={"maxiter": 2000, "ftol": 1e-12})
    assert ref.success
    assert fit.objective == pytest.approx(ref.fun, abs=1e-5)
    assert fit.converged


def test_duplicate_features_are_explicitly_ambiguous():
    d, _ = make_fixture("T8", seed=4)
    fit = fit_weighted_en(d, l1=.005, l2=0., weights=np.ones(4))
    assert np.isfinite(fit.beta).all()
    assert fit.fallback_reason is not None


def test_penalties_bounded_and_weak_term_not_hard_deleted():
    d, _ = make_fixture("T3", seed=0)
    diag = estimate_diagnostics(d)
    for strategy in ("uniform", "instability", "identifiability", "hybrid", "adaptive_lasso"):
        w = build_penalty_weights(diag, strategy=strategy, gamma=1., bounds=(.25, 4.))
        assert (w >= .25).all() and (w <= 4).all()
        assert np.isfinite(w).all()
    assert np.array_equal(build_penalty_weights(diag, "uniform", 1., (.25, 4.)), np.ones(4))
    assert diag.beta[1] != 0


def test_refit_zeroes_unselected_terms_and_matches_ols():
    d, _ = make_fixture("T1", seed=2)
    selected = np.array([True, True, False, False])
    beta = post_refit(d, selected)
    np.testing.assert_allclose(beta[:2], np.linalg.lstsq(d.X[:, :2], d.y, rcond=None)[0], atol=1e-12)
    np.testing.assert_array_equal(beta[2:], np.zeros(2))


def test_invalid_penalties_rejected():
    d, _ = make_fixture("T1")
    with pytest.raises(ValueError, match="penalty"):
        fit_weighted_en(d, l1=-.1, l2=0., weights=np.ones(4))
