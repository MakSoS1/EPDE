"""Offline CatBoost fitting and leave-one-equation-family-out evaluation.

Reference metrics label historical synthetic experiments only. They are never
features. Every seed/noise/configuration of an equation family stays in one fold.
"""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from .candidates import VECTOR_NAMES, VERSION as PORTFOLIO_VERSION
from .features import VERSION as FEATURE_VERSION
from .trees import export_numeric, predict

ASSET_SCHEMA='catboost-advisor/1'


def family_folds(rows):
    for family in sorted({r['family'] for r in rows}):
        yield ([i for i,r in enumerate(rows) if r['family']!=family],
               [i for i,r in enumerate(rows) if r['family']==family])


def quality_loss(record):
    """Declared structural/coefficient loss; unavailable is None, not perfect.

    log(1+selected HD) + .25 log(1+selected RE). This is a benchmark loss
    against the known law, not a forecast R² or a physical-error probability.
    """
    m=record.get('metrics') or {}
    if (record.get('status')!='ok' or not record.get('selected') or m.get('scoring_note')
            or (record.get('baseline') or {}).get('scoring_note')):
        return None
    h,re=m.get('hamming_selected'),m.get('re_sum')
    if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or x<0 for x in (h,re)):
        return None
    return math.log1p(h)+.25*math.log1p(min(re,1e6))


def implementation_hash():
    root=Path(__file__).parent
    # Frozen descriptors/candidates/evaluator/selection rules define the asset ABI.
    return hashlib.sha256(b''.join((root/f).read_bytes().replace(b'\r\n',b'\n')
                                  for f in ('features.py','candidates.py','trees.py','inference.py'))).hexdigest()


def _fit(rows, directory):
    from catboost import CatBoostRegressor
    valid=[r for r in rows if r.get('loss') is not None]
    if len(valid)<8 or len({r['family'] for r in valid})<2:
        return None
    x=np.array([r['vector'] for r in valid]);y=np.array([r['loss'] for r in valid])
    models=[]
    for seed in (17,41,83):
        model=CatBoostRegressor(iterations=160,depth=4,learning_rate=.06,
                               l2_leaf_reg=5,random_seed=seed,verbose=False,
                               thread_count=1,allow_writing_files=False,
                               loss_function='RMSE').fit(x,y)
        compact=export_numeric(model,directory/f'tree-{seed}.json')
        np.testing.assert_allclose(predict(compact,x),model.predict(x),rtol=1e-10,atol=1e-10)
        models.append(compact)
    timed=[r for r in rows if r.get('seconds',0)>0 and math.isfinite(r['seconds'])]
    model=CatBoostRegressor(iterations=100,depth=3,learning_rate=.06,l2_leaf_reg=5,
                           random_seed=17,thread_count=1,verbose=False,
                           allow_writing_files=False).fit([r['vector'] for r in timed],
                                                        np.log1p([r['seconds'] for r in timed]))
    cost=export_numeric(model,directory/'runtime.json')
    # Operational availability is separate from scientific quality.
    from catboost import CatBoostClassifier
    labels=[int(r.get('loss') is not None) for r in rows]
    availability=None
    if len(set(labels))>1:
        classifier=CatBoostClassifier(iterations=80,depth=3,learning_rate=.06,l2_leaf_reg=5,
                                      random_seed=17,thread_count=1,verbose=False,
                                      allow_writing_files=False).fit([r['vector'] for r in rows],labels)
        availability=export_numeric(classifier,directory/'availability.json')
    return {'quality':models,'runtime':cost,'availability':availability,
            'availability_constant':float(np.mean(labels))}


def rank_vectors(models, vectors):
    forecasts=np.array([predict(m,vectors) for m in models['quality']])
    mean=np.maximum(0.,np.mean(forecasts,axis=0));spread=np.std(forecasts,axis=0)
    times=np.maximum(0.,np.expm1(np.clip(predict(models['runtime'],vectors),-20,20)))
    availability=(1/(1+np.exp(-np.clip(predict(models['availability'],vectors),-30,30)))
                  if models.get('availability') else np.full(len(vectors),models['availability_constant']))
    return mean,spread,times,availability


def select_index(models, vectors, rule_index=0, budget_seconds=120.):
    mean,spread,times,availability=rank_vectors(models,vectors)
    eligible=[i for i in range(len(vectors)) if availability[i]>=.5 and times[i]<=budget_seconds]
    chosen=min(eligible,key=lambda i:(mean[i]+.5*spread[i],times[i])) if eligible else rule_index
    # Do not overwrite a baseline for a negligible or unstable prediction.
    if chosen!=rule_index and mean[rule_index]-mean[chosen] <= max(.025,spread[rule_index]+spread[chosen]):
        chosen=rule_index
    return chosen,(mean,spread,times,availability)


def _bootstrap_family_interval(pairs):
    groups=defaultdict(list)
    for p in pairs:
        if p['baseline_loss'] is not None and p['selected_loss'] is not None:
            groups[p['family']].append(p['baseline_loss']-p['selected_loss'])
    values=np.array([np.mean(v) for v in groups.values()])
    if len(values)<3:return None
    rng=np.random.default_rng(20261103)
    boot=np.mean(rng.choice(values,size=(2000,len(values)),replace=True),axis=1)
    return [float(x) for x in np.quantile(boot,[.025,.975])]


def train_bundle(rows, out_dir, provenance):
    """Return a frozen model bundle and auditable OOF choices for every input."""
    from epde_bench.identity import _source_hash
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    model_dir=out_dir/'training-temp';model_dir.mkdir(exist_ok=True)
    reports={};models={};all_pairs=[]
    for method in sorted({r['method'] for r in rows}):
        current=[r for r in rows if r['method']==method]
        pairs=[]
        for train,test in family_folds(current):
            fitted=_fit([current[i] for i in train],model_dir)
            groups=defaultdict(list)
            for i in test:groups[current[i]['input_id']].append(current[i])
            for group in groups.values():
                group=sorted(group,key=lambda r:(r['policy']!='rule',r['policy']))
                if not group or group[0]['policy']!='rule':continue
                chosen=0 if fitted is None else select_index(fitted,[r['vector'] for r in group])[0]
                base,selected=group[0],group[chosen]
                pair={'method':method,'family':base['family'],'dataset':base['dataset'],
                      'noise':base['noise'],'seed':base['seed'],'input_id':base['input_id'],
                      'baseline_policy':'rule','selected_policy':selected['policy'],
                      'baseline_loss':base.get('loss'),'selected_loss':selected.get('loss'),
                      'baseline_seconds':base['seconds'],'selected_seconds':selected['seconds'],
                      'baseline_status':base['status'],'selected_status':selected['status'],
                      'selected_run_id':selected['run_id'],'baseline_run_id':base['run_id'],
                      'train_families':sorted({current[i]['family'] for i in train})}
                assert pair['family'] not in pair['train_families']
                pairs.append(pair)
        comparable=[p for p in pairs if p['baseline_loss'] is not None and p['selected_loss'] is not None]
        deltas=[p['baseline_loss']-p['selected_loss'] for p in comparable]
        mean=float(np.mean(deltas)) if deltas else None
        covered=sum(p['selected_loss'] is not None for p in pairs)
        baseline_covered=sum(p['baseline_loss'] is not None for p in pairs)
        families=len({p['family'] for p in comparable})
        interval=_bootstrap_family_interval(pairs)
        # A modest empirical gate; interval explicitly distinguishes exploratory
        # evidence from a statistically established improvement.
        enabled=bool(mean is not None and mean>.01 and covered>=baseline_covered and families>=4)
        report={'protocol':'leave-one-equation-family-out','inputs':len(pairs),
                'families':families,'scored_pairs':len(comparable),'selected_scored':covered,
                'baseline_scored':baseline_covered,'mean_loss_reduction':mean,
                'family_bootstrap_95_interval':interval,'enabled':enabled,
                'wins':sum(d>1e-9 for d in deltas),'ties':sum(abs(d)<=1e-9 for d in deltas),
                'losses':sum(d< -1e-9 for d in deltas),
                'evidence':'positive grouped interval' if interval and interval[0]>0 else 'exploratory; interval includes zero',
                'reason':('Paired family-held-out average improves without reduced scoring coverage.' if enabled else
                          'No sufficient held-out advantage over Suggested v1; keep the rule baseline.'),
                'runs':len(current),'scored_runs':sum(r.get('loss') is not None for r in current)}
        reports[method]=report;all_pairs+=pairs
        fitted=_fit(current,model_dir)
        if fitted:
            data=np.array([r['vector'] for r in current])
            fitted.update(enabled=enabled,report=report,
                          semantic_signatures=sorted({r['semantic_signature'] for r in current}),
                          domain={'axes':sorted(set(data[:,0].tolist())),
                                  'variables':sorted(set(data[:,1].tolist())),
                                  'log_cells_min':float(data[:,2].min()),'log_cells_max':float(data[:,2].max()),
                                  'noise_max':max(r['noise'] for r in current),
                                  'kinds':sorted({r['kind'] for r in current})})
            models[method]=fitted
    bundle={'schema':ASSET_SCHEMA,'feature_version':FEATURE_VERSION,
            'portfolio_version':PORTFOLIO_VERSION,'feature_names':list(VECTOR_NAMES),
            'implementation_sha256':implementation_hash(),'scientific_source_sha256':_source_hash(),
            'created_at':datetime.now(timezone.utc).isoformat(),'provenance':provenance,
            'target':'log(1 + selected HD) + 0.25 * log(1 + min(selected RE, 1e6))',
            'baseline':'Suggested v1 at the same initial training/search budget',
            'methods':models,'reports':reports}
    return bundle,all_pairs
