import numpy as np
import pytest

from projects.nir1_stability.nir1.design import ResearchDesign
from projects.nir1_stability.nir1.fixtures import make_fixture
from projects.nir1_stability.nir1.performance import (
    compare_reference_fast, fit_with_exact_cache, make_cache_key,
)
from projects.nir1_stability.nir1.regularizers import fit_weighted_en


def changed_design(d, X=None, w=None, names=None, env=None, metadata=None):
    return ResearchDesign(d.X if X is None else X, d.y,
                          d.sample_weight if w is None else w,
                          d.environment_id if env is None else env,
                          d.token_names if names is None else names,
                          d.metadata if metadata is None else metadata)


def test_content_cache_key_distinguishes_values_weights_order_dtype_and_revision():
    d, _ = make_fixture("T1", seed=0)
    key = make_cache_key(d, target_id="y", dtype="float64", revision="a")
    assert key == make_cache_key(changed_design(d), target_id="y", dtype="float64", revision="a")
    X = d.X.copy()
    X[0, 0] += .0001
    assert key != make_cache_key(changed_design(d, X=X), target_id="y", dtype="float64", revision="a")
    w = d.sample_weight.copy()
    w[0] *= 2
    assert key != make_cache_key(changed_design(d, w=w), target_id="y", dtype="float64", revision="a")
    assert key != make_cache_key(changed_design(d, X=d.X[:, ::-1], names=d.token_names[::-1]), target_id="y", dtype="float64", revision="a")
    assert key != make_cache_key(changed_design(d, metadata={**d.metadata, "deriv": "fd"}), target_id="y", dtype="float64", revision="a")
    assert key != make_cache_key(d, target_id="y2", dtype="float64", revision="a")
    assert key != make_cache_key(d, target_id="y", dtype="float32", revision="a")
    assert key != make_cache_key(d, target_id="y", dtype="float64", revision="b")


def test_cached_subset_gram_is_equal_to_fresh_reference():
    d, _ = make_fixture("T3", seed=11)
    support = np.array([True, True, False, True])
    sub = changed_design(d, X=d.X[:, support], names=tuple(np.array(d.token_names)[support]))
    ref = fit_weighted_en(sub, l1=.008, l2=.01, weights=np.ones(support.sum()))
    cache = {}
    fast = fit_with_exact_cache(d, support, cache, l1=.008, l2=.01,
                                weights=np.ones(support.sum()))
    np.testing.assert_allclose(fast.beta[support], ref.beta, rtol=1e-9, atol=1e-11)
    assert np.array_equal(fast.support[support], ref.support)
    # Subset slicing changes BLAS reduction order at ~1e-15: not bitwise equal.
    assert fast.objective == pytest.approx(ref.objective, rel=1e-12, abs=1e-12)
    assert len(cache) == 1
    again = fit_with_exact_cache(d, support, cache, l1=.008, l2=.01,
                                 weights=np.ones(support.sum()))
    np.testing.assert_array_equal(fast.beta, again.beta)


def test_corrupted_cache_does_not_silently_produce_wrong_support():
    d, _ = make_fixture("T1", seed=1)
    cache = {}
    support = np.array([True, True, True, False])
    fit_with_exact_cache(d, support, cache, l1=.01, l2=.01, weights=np.ones(3))
    entry = next(iter(cache.values()))
    entry["gram"].G[0, 0] *= 100
    repaired = fit_with_exact_cache(d, support, cache, l1=.01, l2=.01,
                                    weights=np.ones(3))
    assert repaired.fallback_reason is not None and "cache_corrupt" in repaired.fallback_reason


def test_degenerate_case_triggers_reference_fallback():
    d, _ = make_fixture("T8", seed=1)
    support = np.ones(d.X.shape[1], dtype=bool)
    fit = fit_with_exact_cache(d, support, {}, l1=.01, l2=.0, weights=np.ones(4))
    assert fit.fallback_reason is not None
    assert "reference" in fit.fallback_reason


def test_parity_report_labels_exact_and_lists_fallbacks():
    cases = [make_fixture(name, seed=4)[0] for name in ("T1", "T2", "T8")]
    records = compare_reference_fast(cases)
    assert len(records) == 3
    assert all(row["approximate"] is False for row in records)
    assert all(row["support_equal"] for row in records)
    assert all(row["quality_changed"] is False for row in records)
