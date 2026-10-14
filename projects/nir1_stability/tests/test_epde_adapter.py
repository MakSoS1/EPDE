import numpy as np
import pytest

from projects.nir1_stability.nir1.epde_adapter import resolve_research_variant, run_nir1_epde


def test_production_and_four_factorial_variants_are_separate():
    baseline = resolve_research_variant('default')
    assert baseline == {'instability_metric': 'chi2', 'sparsity_cls': 'vwsr'}
    criterion = resolve_research_variant('nir1_criterion_only')
    regulator = resolve_research_variant('nir1_regulator_only')
    combined = resolve_research_variant('nir1_combined')
    assert criterion['research_objective_metric'] == 'nir1_excess'
    assert criterion.get('research_regularizer_metric') is None
    assert regulator['research_regularizer_metric'] == 'nir1_excess'
    assert regulator.get('research_objective_metric') is None
    assert combined['research_objective_metric'] == combined['research_regularizer_metric'] == 'nir1_excess'
    assert regulator['sparsity_cls'] == combined['sparsity_cls'] == 'nir1_adaptive'
    with pytest.raises(ValueError):
        resolve_research_variant('not_supported')


def test_new_metric_finite_and_identifiability_diagnostic_flags_copies():
    from epde.operators.common.survival import nir1_excess_scores
    rng = np.random.default_rng(7)
    a = rng.normal(size=320)
    b = rng.normal(size=320)
    y = 2 * a - 0.5 * b + rng.normal(0, 0.1, size=320)
    independent = nir1_excess_scores(np.column_stack((a, b)), y, None, (320,), True)
    duplicate = nir1_excess_scores(np.column_stack((a, a + b * 1e-5)), y, None, (320,), True)
    assert independent.shape == duplicate.shape == (2,)
    assert np.all(np.isfinite(independent)) and np.all((independent >= 0) & (independent <= 1))
    assert min(duplicate) > min(independent)
    explicit = nir1_excess_scores(np.column_stack((a, b, np.ones(len(a)))),
                                  y, None, (320,), False)
    assert explicit[:2] == pytest.approx(independent, abs=1e-9)


def test_metric_overrides_are_opt_in_and_production_default_unchanged():
    from epde.interface.search_config import ObjectivesConfig, load_search_config, resolve_sparsity
    baseline = ObjectivesConfig()
    assert baseline.instability_metric == 'chi2'
    assert baseline.research_objective_metric is None
    assert baseline.research_regularizer_metric is None
    assert baseline.objective_metric == baseline.regularizer_metric == 'chi2'
    assert baseline.gram_mode is None
    research = ObjectivesConfig(research_objective_metric='nir1_excess',
                                research_regularizer_metric='cv')
    assert research.objective_metric == 'nir1_excess'
    assert research.regularizer_metric == 'cv'
    assert research.gram_mode == 'axis'
    assert resolve_sparsity('nir1_adaptive').__name__ == 'Nir1AdaptiveSparsity'


def test_factorial_separation_changes_only_its_named_channel():
    from epde.interface.search_config import ObjectivesConfig
    criterion = ObjectivesConfig(**resolve_research_variant('nir1_criterion_only'))
    regulator = ObjectivesConfig(**resolve_research_variant('nir1_regulator_only'))
    combined = ObjectivesConfig(**resolve_research_variant('nir1_combined'))
    assert criterion.regularizer_metric == 'chi2'
    assert criterion.objective_metric == 'nir1_excess'
    assert regulator.objective_metric == 'chi2'
    assert regulator.regularizer_metric == 'nir1_excess'
    assert combined.objective_metric == combined.regularizer_metric == 'nir1_excess'


def test_full_search_adapter_reuses_pic_and_selects_without_truth(monkeypatch):
    from projects.pic.epde_bench import runner
    passed = {}

    def mocked_run_one(dataset, variant, noise, seed, overrides):
        passed.update(dataset=dataset, variant=variant, noise=noise, seed=seed,
                      overrides=overrides)
        return {"status": "ok", "front": [["false"], ["truth"]],
                "objectives": [[4.0, 3.0], [1.0, 1.0]],
                "problem": {"truth": ["false"]}}

    monkeypatch.setattr(runner, "run_one", mocked_run_one)
    record = run_nir1_epde("ode", "nir1_combined", 8, data_seed=3,
                           overrides={"search": {"evolution": {"training_epochs": 2}}})
    assert passed["variant"] == "default"
    assert passed["seed"] == 8
    assert passed["overrides"]["nir1_data_seed"] == 3
    assert passed["overrides"]["search"]["objectives"]["research_regularizer_metric"] == "nir1_excess"
    assert passed["overrides"]["search"]["evolution"]["training_epochs"] == 2
    assert record["research_selected_index"] == 1
    assert record["research_selected"] == ["truth"]


def test_cli_epde_writes_raw_real_search_record(tmp_path, monkeypatch):
    import json
    from projects.nir1_stability.nir1 import epde_adapter
    from projects.nir1_stability.nir1.cli import main

    def fake_search(*args, **kwargs):
        assert kwargs["overrides"]["search"]["evolution"] == {
            "population_size": 4, "training_epochs": 1}
        return {"status": "ok", "front": [["u_t=u"]], "research_selected_index": 0}

    monkeypatch.setattr(epde_adapter, "run_nir1_epde", fake_search)
    output = tmp_path / "real-epde.json"
    assert main(["epde", "--dataset", "ode", "--variant", "nir1_combined",
                 "--seed", "2", "--smoke", "--output", str(output)]) == 0
    assert json.loads(output.read_text())["research_selected_index"] == 0


def test_sparsefront_selects_parsimonious_compromise_without_truth_labels():
    from projects.nir1_stability.nir1.selectors import select_sparsefront
    front = [
        ["0.001 * d^4u/dx1^4{power: 1.0} + 0.1 * d^2u/dx1^2{power: 1.0} = du/dx0{power: 1.0}"],
        ["0.1 * d^2u/dx1^2{power: 1.0} = du/dx0{power: 1.0}"],
    ]
    objectives = [[0.001, 0.02], [0.002, 0.04]]
    correct = select_sparsefront(front, objectives)
    assert correct == 1
    assert select_sparsefront(front, objectives, truth_labels=[False, True]) == correct
    assert select_sparsefront(front, objectives, truth_labels=[True, False]) == correct
    assert select_sparsefront([], []) is None


def test_sparsefront_keeps_original_pic_verdict_without_target_leak(monkeypatch):
    from projects.nir1_stability.nir1.selectors import select_sparsefront
    from projects.pic.epde_bench import runner, datasets
    from types import SimpleNamespace
    front = [
        ["0.001 * d^4u/dx1^4{power: 1.0} + 0.1 * d^2u/dx1^2{power: 1.0} = du/dx0{power: 1.0}"],
        ["0.1 * d^2u/dx1^2{power: 1.0} = du/dx0{power: 1.0}"],
    ]
    metrics = {"success_selected": False, "hamming_selected": 1, "selected_index": 0,
               "success_front": True}
    monkeypatch.setattr(runner, "run_one", lambda *a, **k: {
        "status": "ok", "front": front, "objectives": [[0.001, 0.02], [0.002, 0.04]],
        "metrics": metrics.copy(), "config": {"loader": {}}, "selected": front[0]})
    truth = front[1]
    monkeypatch.setattr(datasets, "load", lambda *a, **k: SimpleNamespace(truth_systems=[truth]))
    record = run_nir1_epde("burgers", "nir1_sparsefront", 7)
    assert record["metrics_pic_original"]["success_selected"] is False
    assert record["metrics"]["success_selected"] is True
    assert record["metrics"]["selected_index"] == 1
    assert record["research_selected_index"] == select_sparsefront(
        front, [[0.001, 0.02], [0.002, 0.04]])
    assert record["selected"] == truth


def test_guarded_penalty_never_increases_penalty_for_correlated_terms():
    from epde.operators.common.survival import nir1_excess_scores, nir1_protected_scores
    rng = np.random.default_rng(11)
    x = rng.normal(size=300)
    z = x + 1e-4 * rng.normal(size=300)
    y = 2*x + 0.1*rng.normal(size=300)
    X = np.column_stack([x, z])
    original = nir1_excess_scores(X, y, None, (300,), True)
    guarded = nir1_protected_scores(X, y, None, (300,), True)
    assert np.isfinite(original).all() and np.isfinite(guarded).all()
    assert guarded.shape == original.shape
    assert np.all((guarded >= 0) & (guarded <= 1))
    assert np.all(guarded <= original + 1e-12)
    assert np.max(guarded) < np.max(original)
    assert resolve_research_variant("nir1_protected_regulator") == {
        "instability_metric": "chi2", "sparsity_cls": "nir1_adaptive",
        "research_regularizer_metric": "nir1_protected",
    }
