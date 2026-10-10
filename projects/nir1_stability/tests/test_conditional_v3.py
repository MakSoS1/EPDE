"""Research-only v3: data-only conditional deletion evidence, not oracle truth."""

import numpy as np
import pytest

from epde.operators.common.survival import (
    nir1_conditional_scores, nir1_protected_scores, block_gram_partition
)
from epde.operators.common.sparsity import instability_scores
from projects.nir1_stability.nir1.epde_adapter import resolve_research_variant


def _sample(seed=108, *, n=880):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    X[:, 1] = .6 * X[:, 0] + .8 * X[:, 1]
    drift = np.linspace(.7, 1.5, n)
    y = drift * X[:, 0] - .7 * X[:, 2] + .2 + .15*rng.normal(size=n)
    return X, y, rng.uniform(.5, 1.5, n)


def test_conditional_no_more_aggressive_than_guarded_and_signal_protection():
    X, y, w = _sample()
    old = nir1_protected_scores(X, y, w, (44, 20))
    changed = nir1_conditional_scores(X, y, w, (44, 20))
    assert np.all(np.isfinite(old)) and np.all(np.isfinite(changed))
    assert np.all((0 <= changed) & (changed <= old + 1e-12))
    # The strong but heterogeneous first coefficient is protected.
    assert old[0] > 0
    assert changed[0] < old[0]


@pytest.mark.parametrize("with_intercept", [True, False])
def test_v3_cached_blocks_match_uncached(with_intercept):
    X, y, w = _sample()
    n, p = X.shape
    blocks = block_gram_partition(X, y, w, (44, 20), 16,
                                  return_yy=True)
    mask = np.array([True, False, True, True, with_intercept], dtype=bool)
    kwargs = dict(metric="nir1_conditional", X=X, y=y, sw=w,
                  grid_shape=(44, 20), active_mask=mask, n_features=p)
    direct = instability_scores(**kwargs)
    reuse = instability_scores(**kwargs, nir1_full_blocks=blocks)
    np.testing.assert_allclose(reuse, direct, rtol=2e-6, atol=2e-8)


def test_v3_scaling_invariance_on_well_conditioned_design():
    X, y, w = _sample(seed=73)
    base = nir1_conditional_scores(X, y, w, (44, 20))
    scaled = X * np.array([1e-2, 3., 50., 1.3])
    alt = nir1_conditional_scores(scaled, y, w, (44, 20))
    np.testing.assert_allclose(alt, base, atol=2e-7, rtol=2e-6)


def test_v3_opt_in_does_not_change_baseline_configuration():
    expected = resolve_research_variant("default")
    conditional = resolve_research_variant("nir1_conditional_regulator")
    assert expected == {"instability_metric": "chi2", "sparsity_cls": "vwsr"}
    assert conditional == {"instability_metric": "chi2",
                           "sparsity_cls": "nir1_adaptive",
                           "research_regularizer_metric": "nir1_conditional"}
