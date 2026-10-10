import numpy as np
import pytest

from projects.nir1_stability.nir1.design import ResearchDesign
from projects.nir1_stability.nir1.diagnostics import estimate_diagnostics
from projects.nir1_stability.nir1.fixtures import make_fixture
from projects.nir1_stability.nir1.model_scores import score_equation


def test_permutation_equivariance_of_scores():
    d, _ = make_fixture("T1", seed=8)
    before = estimate_diagnostics(d)
    order = [2, 0, 3, 1]
    permuted = ResearchDesign(d.X[:, order], d.y, d.sample_weight, d.environment_id,
                              tuple(d.token_names[i] for i in order), d.metadata)
    after = estimate_diagnostics(permuted)
    np.testing.assert_allclose(after.identifiability, before.identifiability[order], atol=1e-9)
    np.testing.assert_allclose(after.excess_variation, before.excess_variation[order], atol=1e-9)
    assert score_equation(after) == pytest.approx(score_equation(before), rel=1e-7)


def test_feature_scale_preserves_dimensionless_diagnostic():
    d, _ = make_fixture("T3", seed=0)
    base = estimate_diagnostics(d)
    x = d.X.copy()
    x[:, 1] *= 99.
    altered = ResearchDesign(x, d.y, d.sample_weight, d.environment_id, d.token_names, d.metadata)
    changed = estimate_diagnostics(altered)
    np.testing.assert_allclose(base.identifiability, changed.identifiability, atol=1e-8)
    np.testing.assert_allclose(base.excess_variation, changed.excess_variation, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(base.beta[1], changed.beta[1] * 99, rtol=1e-5)


def test_duplicate_column_is_not_declared_identified():
    d, _ = make_fixture("T8", seed=13)
    result = estimate_diagnostics(d)
    assert not result.reliable[0] and not result.reliable[1]
    assert result.identifiability[0] < 1e-8
    assert result.identifiability[1] < 1e-8
    assert result.rank < d.X.shape[1]
    assert "rank_deficient" in result.warnings


def test_errors_in_variables_disable_calibrated_uncertainty():
    d, _ = make_fixture("T7", seed=15)
    result = estimate_diagnostics(d)
    assert "errors_in_variables" in result.warnings
    assert not result.reliable.any()
    assert np.isfinite(result.expected_variance).all()


def test_variable_coefficients_are_flagged_not_passed_as_constant_law():
    d, _ = make_fixture("T6", seed=2)
    result = estimate_diagnostics(d)
    assert "variable_coefficients" in result.warnings
    assert np.max(result.excess_variation) > 0


def test_orthogonal_reference_uncertainty_and_scores_are_finite():
    d, _ = make_fixture("T1", seed=3)
    result = estimate_diagnostics(d)
    assert result.rank == 4
    assert np.all(result.identifiability > .9)
    assert np.isfinite(result.expected_variance).all()
    assert np.all(result.expected_variance >= 0)
    assert np.isfinite(score_equation(result))
