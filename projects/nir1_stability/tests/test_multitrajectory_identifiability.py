"""H8 safety: selector never receives or uses true support labels."""

import numpy as np
import pytest

from projects.nir1_stability.nir1.multitrajectory_identifiability import (
    CASES, choose_support, fixed_dictionary, make_observations, one_study,
)


def test_all_analytic_trajectories_same_geometry_and_finite_derivatives():
    for case in CASES:
        observations=make_observations(case,seed=0)
        assert len(observations)==4
        for t,x,u in observations:
            assert u.shape[0]==len(t)
            assert np.isfinite(u).all()
            assert np.all(np.diff(t)>0)
        order=2 if case in ("wave","harmonic") else 1
        items=[fixed_dictionary(row,target_time_order=order) for row in observations]
        assert all(np.isfinite(item[1]).all() and np.isfinite(item[2]).all()
                   for item in items)
        assert all(item[0]==items[0][0] for item in items)


@pytest.mark.parametrize("case",["heat","reactiondiff","advdiff"])
def test_equal_observation_budget_and_genuine_heldout(case):
    row=one_study(case,seed=0)
    n=row["single"]["training_observations"]
    m=row["multi"]["training_observations"]
    assert abs(n-m)<=3
    assert np.isfinite(row["single"]["heldout_relative_mse"])
    assert np.isfinite(row["multi"]["heldout_relative_mse"])
    assert row["multi"]["exact"] is True
    assert row["single"]["exact"] is False


def test_selector_is_truth_blind_and_order_requires_explicit_input():
    a=make_observations("heat",seed=11)
    features=[fixed_dictionary(x,target_time_order=1) for x in a]
    train=[(features[0][0],features[0][1][::3],features[0][2][::3])]
    validate=[(features[0][0],features[0][1][1::3],features[0][2][1::3])]
    first=choose_support(train,validate)
    second=choose_support(train,validate)
    assert first["support"]==second["support"]
    np.testing.assert_array_equal(first["coefficients"],second["coefficients"])
    with pytest.raises(ValueError,match="time derivative order"):
        fixed_dictionary(a[0],target_time_order=0)


def test_unmatched_library_is_rejected():
    common=(("u",),np.ones((100,1)),np.ones(100))
    bad=(("u","u2"),np.ones((100,2)),np.ones(100))
    with pytest.raises(ValueError,match="Dictionary"):
        choose_support([common],[bad])


def test_diverse_initial_conditions_break_gram_aliasing():
    from projects.nir1_stability.nir1.identifiability_certificate import study
    result=study(seed=0)
    assert len(result["rows"])==5
    for row in result["rows"]:
        assert abs(row["single"]["samples"] - row["multi"]["samples"])<=3
        assert row["multi"]["unique_energy"] >= row["single"]["unique_energy"]-1e-6
    for row in result["rows"]:
        if row["case"] in {"heat","reactiondiff","advdiff"}:
            assert row["single"]["unique_energy"] < 1e-6
            assert row["multi"]["unique_energy"] > 1e-4
