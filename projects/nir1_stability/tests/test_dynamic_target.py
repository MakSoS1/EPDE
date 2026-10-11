"""H9 evolution-law restriction: no oracle, no extra search budget."""
import copy
from types import SimpleNamespace

import pytest

from epde.evaluators import simple_function_evaluator
from epde.operators.common.dynamic_target import (
    ensure_temporal_target, is_temporal_target,
)
from projects.nir1_stability.nir1.epde_adapter import (
    resolve_research_variant, run_nir1_epde,
)


def factor(var="u", code=(0,), is_deriv=True):
    return SimpleNamespace(
        variable=var, deriv_code=list(code) if code is not None else None,
        is_deriv=is_deriv,
        evaluator=SimpleNamespace(_evaluator=simple_function_evaluator),
    )


class MockTerm:
    def __init__(self, code=(1,), var="u", extra=False):
        self.structure=[factor(var,code)]
        if extra:
            self.structure.append(factor(var,code,is_deriv=False))
        self.factors_labels=frozenset((var,tuple(code),"extra" if extra else "bare"))
    def contains_deriv(self,var):
        return any(f.variable==var and f.is_deriv and
                   f.deriv_code not in (None,[None]) for f in self.structure)


class MockEq:
    def __init__(self, structure,cap):
        self.structure=structure
        self.main_var_to_explain="u"
        self.pool=object()
        self.metaparameters={"max_terms_number":{"value":cap}}
        self.invalidations=0
    def _invalidate_label_cache(self):
        self.invalidations+=1


def test_only_pure_time_derivative_of_explained_state_eligible():
    assert is_temporal_target(MockTerm((0,)),"u")
    assert is_temporal_target(MockTerm((0,0)),"u")
    for term in [MockTerm((1,)),MockTerm((0,),var="v"),
                 MockTerm((0,),extra=True)]:
        assert not is_temporal_target(term,"u")
    assert not is_temporal_target(MockTerm((0,)),"v")


def test_injection_no_extra_terms_when_cap_full(monkeypatch):
    import epde.structure.main_structures as main
    eq=MockEq([MockTerm((1,)),MockTerm((1,1))],cap=2)
    def draw(pool,mandatory_family,max_factors_in_term,create_derivs):
        assert mandatory_family=="u"
        assert max_factors_in_term==1 and create_derivs
        return MockTerm((0,))
    monkeypatch.setattr(main,"Term",draw)
    assert ensure_temporal_target(eq)
    assert len(eq.structure)==2
    assert sum(is_temporal_target(t,"u") for t in eq.structure)==1
    assert eq.invalidations==1
    assert not ensure_temporal_target(eq)


def test_append_target_only_when_below_existing_cap(monkeypatch):
    import epde.structure.main_structures as main
    eq=MockEq([MockTerm((1,))],cap=3)
    monkeypatch.setattr(main,"Term",lambda *a,**k:MockTerm((0,)))
    assert ensure_temporal_target(eq)
    assert len(eq.structure)==2


def test_fail_if_temporal_token_missing_not_silent(monkeypatch):
    import epde.structure.main_structures as main
    eq=MockEq([MockTerm((1,)),MockTerm((2,))],cap=2)
    monkeypatch.setattr(main,"Term",lambda *a,**k:MockTerm((1,)))
    with pytest.raises(RuntimeError,match="could not be generated"):
        ensure_temporal_target(eq,max_attempts=3)


def test_scoped_environment_reverts_on_failure(monkeypatch):
    import os
    import projects.pic.epde_bench.runner as runner
    monkeypatch.setenv("NIR1_DYNAMIC_TARGET","external")
    seen=[]
    def fake_runner(*args,**kwargs):
        seen.append(os.environ.get("NIR1_DYNAMIC_TARGET"))
        return {"status":"error","reason":"injected mock failure"}
    monkeypatch.setattr(runner,"run_one",fake_runner)
    out=run_nir1_epde("nir1_heat","nir1_dynamic_anchor",0)
    assert out["status"]=="error" and seen==["1"]
    assert os.environ["NIR1_DYNAMIC_TARGET"]=="external"
    out=run_nir1_epde("nir1_heat","default",0)
    assert out["status"]=="error" and seen[-1] is None
    assert os.environ["NIR1_DYNAMIC_TARGET"]=="external"


def test_never_change_baseline_search_config_or_evolution_budget():
    assert resolve_research_variant("nir1_dynamic_anchor")==resolve_research_variant("default")
