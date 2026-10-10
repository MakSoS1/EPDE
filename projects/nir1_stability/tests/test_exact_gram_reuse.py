"""E1 optimization: parity with pre-optimization full-data Gram evaluation."""

import numpy as np
import pytest

from epde.operators.common.survival import (
    heterogeneity_scores, nir1_excess_scores, nir1_protected_scores,
)


def _reference(X, y, weights, fit_intercept, grid_shape, *, protect):
    """Frozen formula from original NIR1 version; recomputes full-data Gram."""
    n, p = X.shape
    excess = heterogeneity_scores(X, y, weights, grid_shape,
                                  fit_intercept=fit_intercept)
    A = np.column_stack((X, np.ones(n))) if fit_intercept else X
    gram = A.T @ (weights[:, None] * A)
    norm = np.sqrt(np.maximum(np.diag(gram), 0))
    safe = np.where(norm > 0, norm, 1)
    scaled = gram / np.outer(safe, safe)
    q = np.zeros(p)
    for j in range(p):
        if norm[j] == 0:
            continue
        other = np.arange(scaled.shape[0]) != j
        if not other.any():
            q[j] = 1.
            continue
        cross = scaled[other, j]
        coef = np.linalg.lstsq(
            scaled[np.ix_(other, other)], cross, rcond=1e-10)[0]
        q[j] = np.clip(1.0 - cross @ coef / scaled[j, j], 0., 1.)
    scores = excess * q if protect else 0.5 * (excess + (1. - q))
    return np.where(norm[:p] > 0, np.clip(scores, 0., 1.), 1.)


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("correlation", [0., 0.85, 0.9999])
def test_block_gram_reuse_matches_legacy_full_matmul(weighted, fit_intercept, correlation):
    rng = np.random.default_rng(20261010)
    n = 880
    X = rng.normal(size=(n, 4))
    X[:, 1] = correlation * X[:, 0] + np.sqrt(1-correlation**2)*X[:, 1]
    if fit_intercept:
        original = X
    else:
        original = np.column_stack([X, np.ones(n)])
    w = rng.uniform(0.5, 1.5, size=n) if weighted else np.ones(n)
    y = 1.3 * X[:, 0] - 0.8 * X[:, 2] + 0.2 + rng.normal(0, .2, n)
    reference = _reference(original, y, w, fit_intercept, (44, 20), protect=False)
    observed = nir1_excess_scores(original, y, w, (44, 20),
                                  fit_intercept=fit_intercept)
    np.testing.assert_allclose(observed, reference, atol=1e-8, rtol=2e-7)
    reference_protect = _reference(original, y, w, fit_intercept, (44, 20), protect=True)
    observed_protect = nir1_protected_scores(original, y, w, (44, 20),
                                            fit_intercept=fit_intercept)
    np.testing.assert_allclose(observed_protect, reference_protect, atol=1e-8,
                               rtol=2e-7)


def test_full_gram_reuse_matches_legacy_gram_in_weighted_coordinates():
    from epde.operators.common.survival import _het_components
    rng = np.random.default_rng(17)
    X = rng.normal(size=(450, 5))
    y = rng.normal(size=450)
    w = np.exp(rng.uniform(-1., 1., 450))
    for intercept in (True, False):
        *values, G = _het_components(
            X, y, w, (45, 10), fit_intercept=intercept,
            return_full_gram=True)
        A = np.column_stack([X, np.ones(len(X))]) if intercept else X
        direct = A.T @ (w[:, None] * A)
        np.testing.assert_allclose(G, direct, atol=1e-10, rtol=1e-12)
        previous = _het_components(X, y, w, (45, 10), fit_intercept=intercept)
        for actual, prior in zip(values, previous):
            np.testing.assert_array_equal(actual, prior)
