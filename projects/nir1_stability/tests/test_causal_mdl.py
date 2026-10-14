"""Research-only H10: same search counts, no hidden oracle, MDL scoring."""
import math
from types import SimpleNamespace

import pytest

from epde.operators.common.causal_mdl import (
    dynamical_description_length, temporal_effective_levels,
)
from projects.nir1_stability.nir1.epde_adapter import (
    resolve_research_variant, run_nir1_epde,
)


def _eq():
    target=SimpleNamespace(structure=[SimpleNamespace(deriv_code=[0])])
    low=SimpleNamespace(structure=[SimpleNamespace(deriv_code=[1,1])])
    high=SimpleNamespace(structure=[SimpleNamespace(deriv_code=[1,1,1,1])])
    # target 0, active low 1 + active high 2, no intercept
    class Model:
        structure=[target, low, high]
        target_idx=0
        weights_internal=[1.,1.,0.]
        def weight_index(self, idx, target):
            return idx-1 if idx>target else idx
    return Model()


def test_effective_count_uses_time_levels_not_spatial_points():
    assert temporal_effective_levels({0:(81,96),1:(61,192)})==61
    with pytest.raises(ValueError):
        temporal_effective_levels({})


def test_complexity_rewards_keeping_low_order_structure():
    q=_eq()
    unit=math.log(81)/81
    # Low derivative 2nd-order has no extra order charge.
    assert dynamical_description_length(q,{0:(81,96)})==pytest.approx(4*unit)
    q.weights_internal=[1.,0.,0.]
    assert dynamical_description_length(q,{0:(81,96)})==pytest.approx(unit)
    q.weights_internal=[0.,1.,0.]
    # Fourth spatial derivative counts base 1 + excess 2.
    assert dynamical_description_length(q,{0:(81,96)})==pytest.approx(3*unit)


def test_mdl_scope_no_leak_into_native_or_existing_h9(monkeypatch):
    import os
    import projects.pic.epde_bench.runner as runner
    seen=[]
    def fake(*a,**k):
        seen.append((os.getenv("NIR1_DYNAMIC_TARGET"),os.getenv("NIR1_DYNAMIC_MDL")))
        return {"status":"error"}
    monkeypatch.setattr(runner,"run_one",fake)
    monkeypatch.setenv("NIR1_DYNAMIC_MDL","old")
    run_nir1_epde("nir1_heat","nir1_dynamic_mdl",0)
    run_nir1_epde("nir1_heat","nir1_dynamic_anchor",0)
    run_nir1_epde("nir1_heat","default",0)
    assert seen==[("1","1"),("1",None),(None,None)]
    assert os.getenv("NIR1_DYNAMIC_MDL")=="old"


def test_h10_has_identical_epde_libraries_and_vwsr():
    assert resolve_research_variant("nir1_dynamic_mdl") == resolve_research_variant("default")
