"""H14 exposed EPDE variant uses SAME H12 evolution, only final decision."""
import os

from projects.nir1_stability.nir1.epde_adapter import (
    resolve_research_variant, run_nir1_epde,
)


def test_closed_variant_has_identical_preprocessing_and_search_budget():
    assert resolve_research_variant("nir1_dynamic_mdl_closed")==resolve_research_variant(
        "nir1_dynamic_mdl_repair")
    assert resolve_research_variant("nir1_dynamic_mdl_closed")==resolve_research_variant(
        "default")


def test_closed_variant_chooses_truthblind_and_restores_flags(monkeypatch):
    import projects.pic.epde_bench.runner as runner
    captures=[]
    false="-0.92 * du/dx0{power: 1.0} * u{power: 3.0} = d^2u/dx0^2{power: 1.0}"
    real="0.8 * u{power: 1.0} + -0.8 * u{power: 2.0} = du/dx0{power: 1.0}"
    def fake(*a,**k):
        captures.append((os.getenv("NIR1_DYNAMIC_TARGET"),
                         os.getenv("NIR1_DYNAMIC_MDL"),
                         os.getenv("NIR1_DYNAMIC_REPAIR")))
        return {"status":"ok", "front":[[false],[real]],
                "objectives":[[.01,0.],[.09,0.]],
                "metrics":{"success_selected":False,"selected_index":0},
                "config":{"loader":{}}, "selected":[false]}
    monkeypatch.setattr(runner,"run_one",fake)
    monkeypatch.setenv("NIR1_DYNAMIC_TARGET","external")
    result=run_nir1_epde("nir1_h11_logistic_field",
                         "nir1_dynamic_mdl_closed",0)
    assert captures==[("1","1","1")]
    assert result["research_selected_index"]==1
    assert result["metrics_pic_original"]["selected_index"]==0
    assert result["metrics"]["selected_index"]==1
    assert result["metrics"]["success_selected"] is True
    assert result["selection_strategy"]=="nir1_h14_causal_normal_form_v1"
    assert os.environ["NIR1_DYNAMIC_TARGET"]=="external"
    assert "NIR1_DYNAMIC_REPAIR" not in os.environ


def test_when_only_implicit_candidates_exist_h14_abstains_as_error(monkeypatch):
    import projects.pic.epde_bench.runner as runner
    false="1.0 * u{power: 1.0} * du/dx0{power: 1.0} = d^2u/dx0^2{power: 1.0}"
    monkeypatch.setattr(runner,"run_one",lambda *a,**k:{
        "status":"ok","front":[[false]],"objectives":[[.001,0.]],
        "metrics":{"success_selected":False,"selected_index":0},
        "config":{"loader":{}},"selected":[false]})
    result=run_nir1_epde("nir1_h11_logistic_field","nir1_dynamic_mdl_closed",0)
    assert result["research_selected_index"] is None
    assert result["metrics"]["success_selected"] is False
