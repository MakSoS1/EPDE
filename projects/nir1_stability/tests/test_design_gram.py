import numpy as np
import pytest

from projects.nir1_stability.nir1.design import ResearchDesign
from projects.nir1_stability.nir1.gram import accumulate_gram, subset_gram


def example(weights=None):
    return ResearchDesign(
        X=np.array([[1., 0.], [1., 1.], [0., 1.]]),
        y=np.array([1., 2., 3.]),
        sample_weight=np.array([1., 2., 3.] if weights is None else weights),
        environment_id=np.array(["b", "a", "b"]),
        token_names=("u", "u_x"), metadata={"source": "test"},
    )


def test_blocked_gram_matches_weighted_moments():
    d = example()
    direct_g = d.X.T @ (d.sample_weight[:, None] * d.X)
    direct_b = d.X.T @ (d.sample_weight * d.y)
    one = accumulate_gram(d, chunk_rows=1)
    all_rows = accumulate_gram(d, chunk_rows=8192)
    for result in (one, all_rows):
        np.testing.assert_allclose(result.G, direct_g, atol=1e-12)
        np.testing.assert_allclose(result.b, direct_b, atol=1e-12)
        assert result.yy == pytest.approx(np.dot(d.sample_weight, d.y**2))
        assert result.weight_sum == pytest.approx(6.0)
        assert list(result.by_environment) == ["a", "b"]
        np.testing.assert_allclose(sum(v[0] for v in result.by_environment.values()), result.G)


def test_subset_gram_preserves_column_order():
    g = accumulate_gram(example())
    selected = subset_gram(g, [1, 0])
    np.testing.assert_array_equal(selected.G, g.G[np.ix_([1, 0], [1, 0])])
    np.testing.assert_array_equal(selected.b, g.b[[1, 0]])
    assert selected.yy == g.yy


@pytest.mark.parametrize("weights", [[-1., 2., 3.], [float("nan"), 2., 3.], [0., 0., 0.]])
def test_invalid_weights_rejected(weights):
    with pytest.raises(ValueError, match="weight"):
        example(weights)


def test_duplicate_tokens_rejected():
    with pytest.raises(ValueError, match="token"):
        ResearchDesign(np.ones((3, 2)), np.ones(3), np.ones(3),
                       np.zeros(3), ("x", "x"), {})


def test_nonfinite_data_rejected():
    d = example()
    x = d.X.copy()
    x[1, 1] = np.inf
    with pytest.raises(ValueError, match="finite"):
        ResearchDesign(x, d.y, d.sample_weight, d.environment_id, d.token_names, {})


def test_shape_mismatch_rejected():
    d = example()
    with pytest.raises(ValueError, match="shape"):
        ResearchDesign(d.X, d.y[:-1], d.sample_weight, d.environment_id, d.token_names, {})


def test_invalid_chunk_size_rejected():
    with pytest.raises(ValueError, match="chunk"):
        accumulate_gram(example(), chunk_rows=0)
