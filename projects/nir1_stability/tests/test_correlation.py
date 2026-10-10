import numpy as np

from projects.nir1_stability.nir1.correlation import correlation_groups, swap_refine
from projects.nir1_stability.nir1.design import ResearchDesign
from projects.nir1_stability.nir1.diagnostics import estimate_diagnostics
from projects.nir1_stability.nir1.fixtures import make_fixture
from projects.nir1_stability.nir1.historical import historical_vclog_weights
from projects.nir1_stability.nir1.regularizers import SparseFit


def test_correlated_pair_clusters_are_nonoverlapping_and_deterministic():
    d, _ = make_fixture("T2", seed=0, correlation=.999)
    groups = correlation_groups(d, threshold=.9)
    assert any(0 in group and 1 in group for group in groups)
    assert sorted(k for group in groups for k in group) == list(range(d.X.shape[1]))


def test_orthogonal_design_leaves_unique_support_unchanged():
    d, _ = make_fixture("T1", seed=4)
    initial = SparseFit(np.array([1.5, -.7, 0., 0.]),
                        np.array([1, 1, 0, 0], dtype=bool), 1., True, 0., None, 1)
    assert np.array_equal(swap_refine(d, initial).support, initial.support)


def test_swap_recovers_missing_correlated_truth_without_access_to_oracle():
    rng = np.random.default_rng(10)
    x = rng.normal(size=350)
    decoy = x + .15 * rng.normal(size=350)
    X = np.column_stack([x, decoy, rng.normal(size=350)])
    y = 1.2 * x + .03 * rng.normal(size=350)
    d = ResearchDesign(X, y, np.ones(350), np.repeat("a", 350),
                       ("u", "u*x", "u_x"), {})
    start = SparseFit(np.array([0., 1.2, 0.]), np.array([False, True, False]),
                      1., True, 0., None, 1)
    end = swap_refine(d, start)
    assert end.support[0] and not end.support[1]


def test_historical_vclog_proxy_is_finite_and_labeled_as_unreproduced():
    d, _ = make_fixture("T5", seed=2)
    diag = estimate_diagnostics(d)
    weights, metadata = historical_vclog_weights(d, diag)
    assert weights.shape == diag.beta.shape
    assert np.all(np.isfinite(weights))
    assert np.all(weights >= 0)
    assert metadata["historical_status"] == "archival_unreproduced"
    assert "historical_reported_perfect" in metadata
