"""H12 repair is opt-in; H11 immutable, no extra terms/epochs or truth."""
import os

from projects.nir1_stability.nir1.epde_adapter import (
    resolve_research_variant, run_nir1_epde,
)


def test_h12_retains_original_search_budget_and_token_library():
    assert resolve_research_variant("nir1_dynamic_mdl_repair") == (
        resolve_research_variant("default"))


def test_h12_repair_scoped_and_isolated(monkeypatch):
    import projects.pic.epde_bench.runner as runner
    seen=[]
    def stub(*a, **k):
        seen.append({name:os.environ.get(name) for name in (
            "NIR1_DYNAMIC_TARGET", "NIR1_DYNAMIC_MDL",
            "NIR1_DYNAMIC_REPAIR")})
        return {"status":"error","error":"test stub no data"}
    monkeypatch.setattr(runner,"run_one",stub)
    monkeypatch.setenv("NIR1_DYNAMIC_REPAIR","external")
    run_nir1_epde("nir1_heat","nir1_dynamic_mdl_repair",0)
    run_nir1_epde("nir1_heat","nir1_dynamic_mdl_select",0)
    run_nir1_epde("nir1_heat","default",0)
    assert seen==[
        dict(NIR1_DYNAMIC_TARGET="1",NIR1_DYNAMIC_MDL="1",NIR1_DYNAMIC_REPAIR="1"),
        dict(NIR1_DYNAMIC_TARGET="1",NIR1_DYNAMIC_MDL="1",NIR1_DYNAMIC_REPAIR=None),
        dict(NIR1_DYNAMIC_TARGET=None,NIR1_DYNAMIC_MDL=None,NIR1_DYNAMIC_REPAIR=None),
    ]
    assert os.environ.get("NIR1_DYNAMIC_REPAIR")=="external"
