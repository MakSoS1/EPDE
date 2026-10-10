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
