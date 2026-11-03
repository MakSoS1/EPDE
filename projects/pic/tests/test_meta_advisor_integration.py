"""Learned advice is real, bounded and independent of reference-law fields."""
from copy import deepcopy
from pathlib import Path
import tempfile

import numpy as np
import pytest

from support.discovery_lab.meta_advisor import features, trees, training
from support.discovery_lab import starting_settings


def arrays():
    t=np.linspace(0,2,81);x=np.linspace(0,2*np.pi,96)
    return {'axis_0':t,'axis_1':x,'var_u':np.exp(-.2*t[:,None])*np.sin(x[None,:])}


def test_enrichment_reads_signal_but_not_reference_equation():
    a=arrays();context=starting_settings.summarize_input(arrays=a)
    enriched=features.enrich_context(context,arrays=a)
    assert set(enriched['data_features'])==set(features.FEATURE_NAMES)
    a['reference_truth']='do not use this label'
    assert features.enrich_context(context,arrays=a)==enriched
    a['var_u']*=3
    assert features.enrich_context(context,arrays=a)['data_features']!=enriched['data_features']
    assert 'data_features' not in context


def test_heavy_enrichment_never_touches_arrays():
    class Exploding:
        def items(self):raise AssertionError('heavy data read')
    context={'heavy':True}
    assert features.enrich_context(context,arrays=Exploding())==context


def test_numeric_export_matches_native_on_border_neighbours():
    cb=pytest.importorskip('catboost')
    rng=np.random.default_rng(13);x=rng.normal(size=(150,3));y=(x[:,0]>0)+x[:,1]**2
    model=cb.CatBoostRegressor(iterations=20,depth=3,verbose=False,thread_count=1,
                              allow_writing_files=False,random_seed=17).fit(x,y)
    with tempfile.TemporaryDirectory() as td:
        exported=trees.export_numeric(model,Path(td)/'model.json')
    points=[v.copy() for v in x[:10]]
    for tree in exported['trees']:
        for feature,border in tree['splits']:
            for value in (np.nextafter(border,-np.inf),border,np.nextafter(border,np.inf)):
                row=np.zeros(3);row[feature]=value;points.append(row)
    np.testing.assert_allclose(trees.predict(exported,points),model.predict(points),atol=1e-10,rtol=1e-10)


def test_automatic_activation_requires_positive_family_interval():
    assert not training.activation_allowed(.2,32,32,7,[-.03,.4])
    assert not training.activation_allowed(.2,31,32,7,[.01,.4])
    assert not training.activation_allowed(.2,32,32,2,[.01,.4])
    assert training.activation_allowed(.2,32,32,7,[.01,.4])


def test_all_seeds_noise_and_configs_of_family_stay_out_of_training():
    rows=[{'family':f,'seed':s,'noise':n,'policy':p} for f in ('heat','wave','reaction')
          for s in (0,1) for n in (0,2) for p in ('rule','candidate')]
    for train,test in training.family_folds(rows):
        assert set(rows[i]['family'] for i in train).isdisjoint(rows[i]['family'] for i in test)
        assert len(test)==8


def test_suggestion_returns_learned_provenance_even_when_model_declines():
    context=starting_settings.summarize_input(arrays=arrays())
    result=starting_settings.suggest_settings('deepmod',{'max_iterations':5000},context)
    assert result['learned']['engine']=='CatBoost'
    assert result['learned']['applied'] is False


def test_unknown_quality_is_not_zero():
    assert training.quality_loss({'status':'timeout'}) is None
    assert training.quality_loss({'status':'ok','selected':['something'],
        'metrics':{'scoring_note':'outside grammar','hamming_selected':0,'re_sum':0}}) is None


def test_fixed_policy_is_selected_from_training_families_only():
    from support.discovery_lab.meta_advisor.training import best_fixed_policy
    rows=[{'policy':p,'input_id':str(i),'family':'train','loss':v}
          for i in range(3) for p,v in [('rule',2.),('good',1.),('unscored',None)]]
    assert best_fixed_policy(rows)=='good'
    assert best_fixed_policy([{'policy':'rule','input_id':'a','loss':None}])=='rule'


def test_deepmod_architecture_is_part_of_profile_compatibility():
    from support.discovery_lab.meta_advisor.candidates import semantic_signature
    assert semantic_signature('deepmod',{'hidden_layers':[40,40]}) != semantic_signature('deepmod',{'hidden_layers':[100,100]})


def test_full_local_budget_is_not_ranked_by_a_short_budget_model():
    result=starting_settings.suggest_settings('deepmod',{'max_iterations':5000},
        {'heavy':False,'data_features':{'axes':2}},hosted=False)
    assert not result['learned']['applied']
    assert 'local' in result['learned']['reason'].lower()


def test_packaged_model_requires_complete_real_campaign_and_auditable_folds():
    import json
    from support.discovery_lab.meta_advisor.inference import load_bundle
    from support.discovery_lab.meta_advisor.candidates import VECTOR_NAMES
    bundle,digest=load_bundle()
    assert bundle['provenance']['complete']
    assert bundle['provenance']['runs']==512
    assert set(bundle['methods'])=={'epde','pdecomp','deepmod','discover'}
    assert bundle['methods']['deepmod']['enabled']
    assert len(digest)==64
    assert not any('truth' in name or 'dataset' in name for name in VECTOR_NAMES)
    path=Path(__file__).resolve().parents[1]/'experiments/advisor_v1/family-heldout-choices.json'
    pairs=json.loads(path.read_text())
    assert len(pairs)==128
    for p in pairs:
        assert p['family'] not in p['train_families']

def test_packaged_deepmod_model_changes_only_allowed_values_on_real_input():
    import json
    from epde_bench.datasets import load
    from support.discovery_lab.meta_advisor.candidates import changes_for, semantic_signature
    root=Path(__file__).resolve().parents[1]
    record=json.loads((root/'public_results/diffusion-deepmod.json').read_text())
    problem=load(record['dataset']);base=record['baseline']['settings'];before=deepcopy(base)
    context=features.enrich_context(starting_settings.summarize_input(
        record_name=record['dataset'],problem=problem,noise_percent=0),problem=problem)
    report=starting_settings.suggest_settings('deepmod',base,context)
    assert report['learned']['applied']
    assert report['learned']['selected_policy']=='sparse'
    assert len(report['learned']['ranking'])==4
    assert report['settings']['threshold']==.25
    assert semantic_signature('deepmod',base)==semantic_signature('deepmod',report['settings'])
    assert set(changes_for('deepmod',base,report['settings'])) <= {'learning_rate','threshold'}
    assert base==before
