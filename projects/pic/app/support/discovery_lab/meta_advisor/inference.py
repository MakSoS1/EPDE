"""Fast inference from trusted packaged CatBoost assets, with honest fallback."""
from __future__ import annotations
from copy import deepcopy
from functools import lru_cache
import gzip
import hashlib
import json
import math
from pathlib import Path
from .features import VERSION as FEATURE_VERSION
from .candidates import VERSION as PORTFOLIO_VERSION, VECTOR_NAMES, portfolio, vector, semantic_signature

ASSETS=Path(__file__).with_name('assets')


@lru_cache(maxsize=2)
def load_bundle(asset_dir=ASSETS):
    directory=Path(asset_dir)
    manifest=json.loads((directory/'manifest.json').read_text())
    raw=(directory/'model.json.gz').read_bytes()
    if len(raw)>5_000_000 or hashlib.sha256(raw).hexdigest()!=manifest['sha256']:
        raise ValueError('Model checksum mismatch')
    with gzip.GzipFile(fileobj=__import__('io').BytesIO(raw)) as stream:
        decoded=stream.read(10_000_001)
    if len(decoded)>10_000_000:raise ValueError('Model exceeds size limit')
    bundle=json.loads(decoded)
    from .training import implementation_hash
    from epde_bench.identity import _source_hash
    if (bundle.get('schema')!='catboost-advisor/1' or bundle.get('feature_version')!=FEATURE_VERSION
            or bundle.get('portfolio_version')!=PORTFOLIO_VERSION
            or bundle.get('feature_names')!=list(VECTOR_NAMES)
            or bundle.get('implementation_sha256')!=implementation_hash()
            or bundle.get('scientific_source_sha256')!=_source_hash()):
        raise ValueError('Model does not match the running scientific implementation')
    return bundle,manifest['sha256']


def recommend(method, base, context, fallback, *, protocol='native', hosted=True, asset_dir=ASSETS):
    report=deepcopy(fallback)
    learned={'applied':False,'engine':'CatBoost','reason':'No compatible model is available.'}
    report['learned']=learned
    def decline(reason):
        learned['reason']=reason
        report['reasons'].append('CatBoost: '+reason)
        return report
    if protocol!='native':return decline('The learned portfolio covers native protocols only; kept your fixed-protocol settings.')
    if context.get('heavy'):return decline('Large datasets remain metadata-only on the host; kept the full profile.')
    if not context.get('data_features'):return decline('Numerical descriptors are unavailable; using the rule-based starting settings.')
    try:
        bundle,digest=load_bundle(Path(asset_dir))
        model=bundle['methods'].get(method)
        learned.update(model_sha256=digest,training=bundle['provenance'],baseline=bundle['baseline'])
        if not model:return decline('There are not enough scored training runs for this method.')
        learned['validation']=model['report']
        if not model['enabled']:return decline(model['report']['reason'])
        domain=model['domain'];f=context['data_features']
        if (context.get('kind') not in domain['kinds'] or f['axes'] not in domain['axes']
                or f['variables'] not in domain['variables']
                or not domain['log_cells_min']-.7<=f['log_cells']<=domain['log_cells_max']+.7
                or not 0<=context.get('noise_percent',0)<=domain['noise_max']
                or f['uniform_fraction']<1):
            return decline('Input geometry, spacing or declared noise lies outside the evaluated training domain.')
        if semantic_signature(method,base) not in model['semantic_signatures']:
            return decline('This scientific profile or target/library definition was not evaluated; kept it unchanged.')
        candidates=portfolio(method,base,context,protocol=protocol,hosted=hosted)
        if not candidates or candidates[0][0]!='rule':return decline('No safe candidate portfolio fits the existing resource limits.')
        from .training import select_index
        selected,values=select_index(model,[vector(context,method,c[2]) for c in candidates])
        mean,spread,times,availability=values
        learned['ranking']=[{'policy':c[0],'label':c[1],'predicted_loss':float(mean[i]),
                             'model_disagreement':float(spread[i]),'estimated_seconds':float(times[i]),
                             'estimated_scorability':float(availability[i])} for i,c in enumerate(candidates)]
        learned.update(applied=True,selected_policy=candidates[selected][0],
                       reason='Ranked valid configurations using offline-trained CatBoost models.',
                       uncertainty='Model disagreement is not a calibrated confidence interval.')
        report['settings']=deepcopy(candidates[selected][2]);report['version']='catboost-starting-settings-v1'
        report['source']='CatBoost · numerical descriptors + frozen scientific parameter sweeps'
        report['reasons'].append(f"CatBoost selected {candidates[selected][1]}. Predicted benchmark loss {mean[selected]:.3g}; estimated search time {times[selected]:.1f} s on the training runner, not a server time guarantee.")
        report['reasons'].append('The model ranks a finite tested portfolio, not all possible settings. Validate the discovered equation and keep manual adjustments.')
        report['reasons'].append('Family-held-out evidence: '+model['report']['evidence']+'.')
        return report
    except (OSError,ValueError,KeyError,TypeError,OverflowError,IndexError) as exc:
        return decline('Model asset is missing, invalid or incompatible; using the rule-based starting settings.')
