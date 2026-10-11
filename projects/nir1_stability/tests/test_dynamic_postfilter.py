"""H9 physics prior applied at the final front versus during evolution."""
import pytest
from projects.nir1_stability.nir1.dynamic_postfilter import (
    is_explicit_temporal_equation, select_temporal_postfilter,
)

TEMP="0.3 * d^2u/dx1^2{power: 1.0} = du/dx0{power: 1.0}"
SPATIAL="0.4 * du/dx1{power: 1.0} = d^2u/dx1^2{power: 1.0}"
SECOND="0.4 * d^2u/dx1^2{power: 1.0} = d^2u/dx0^2{power: 1.0}"


def test_temporal_target_distinguishes_physical_law_from_spatial_identity():
    assert is_explicit_temporal_equation(TEMP)
    assert is_explicit_temporal_equation(SECOND)
    assert not is_explicit_temporal_equation(SPATIAL)
    assert not is_explicit_temporal_equation("0.1 * du/dx0{power: 1.0} = u{power: 1.0}")
    assert not is_explicit_temporal_equation("u{power: 1.0} = du/dx1{power: 1.0}")


def test_filter_excludes_zero_discrepancy_spatial_identity_with_no_oracle():
    front=[[SPATIAL], [TEMP], [SECOND]]
    objectives=[[0.0,0.0], [0.04,0.5], [0.02,0.5]]
    idx=select_temporal_postfilter(front, objectives)
    assert idx in (1,2)
    assert idx==select_temporal_postfilter(front, objectives, truth_labels=[0,1,0])
    assert idx==select_temporal_postfilter(front, objectives, truth_labels=[1,0,1])
    assert select_temporal_postfilter([[SPATIAL]],[[0.0,0.0]]) is None


def test_filter_requires_every_equation_dynamic_in_coupled_system():
    assert select_temporal_postfilter(
        [[TEMP,SPATIAL],[TEMP,SECOND]],
        [[0.0,0.0,0.0,0.0],[0.1,0.2,0.1,0.2]])==1


def test_front_objective_mismatch_cannot_silently_vote():
    with pytest.raises(ValueError,match="paired"):
        select_temporal_postfilter([[TEMP]],[])
