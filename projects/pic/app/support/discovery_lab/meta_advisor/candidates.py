"""Frozen, editable portfolios; no reference equations or results are consulted."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
from .features import FEATURE_NAMES

VERSION='candidate-portfolio-v1'
PARAMETER_NAMES=('population','epochs','terms','deriv_power','polynomial_preprocessor',
                 'window','threshold','ridge','normalized','iterations','learning_rate',
                 'samples','epsilon','max_length','max_add_count','batch_size')
VECTOR_NAMES=FEATURE_NAMES+('declared_noise',)+PARAMETER_NAMES


def settings_hash(settings):
    return hashlib.sha256(json.dumps(settings,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def vector(context, method, settings):
    features=context.get('data_features') or {}
    if set(features)!=set(FEATURE_NAMES):
        raise ValueError('Numerical descriptors are unavailable or incompatible')
    search=settings.get('search') or {};evolution=search.get('evolution') or {}
    pool=search.get('search_space') or {};prep=search.get('preprocessing') or {}
    optimizer=settings.get('optimizer') or settings.get('direct_optimizer') or {}
    kw=prep.get('preprocessor_kwargs') or {}
    window=kw.get('polynomial_window',0)
    if isinstance(window,(list,tuple)):window=max(window,default=0)
    numbers=[evolution.get('population_size',0),evolution.get('training_epochs',0),
             pool.get('equation_terms_max_number',0),pool.get('deriv_fun_pow',0),
             float(prep.get('default_preprocessor_type')=='poly'),window or 0,
             settings.get('threshold',optimizer.get('threshold',0)),optimizer.get('alpha',0),
             float(bool(optimizer.get('normalize_columns',False))),settings.get('max_iterations',0),
             settings.get('learning_rate',0),settings.get('max_samples',settings.get('n_samples',0)),
             settings.get('epsilon',0),settings.get('max_length',0),settings.get('max_add_count',0),
             settings.get('batch_size',0)]
    result=[float(features[k]) for k in FEATURE_NAMES]+[float(context.get('noise_percent',0))]+list(map(float,numbers))
    if not all(math.isfinite(v) for v in result):raise ValueError('Invalid advisor feature vector')
    return result


def portfolio(method, base, context, *, protocol='native', hosted=True):
    """Four policies at the rule suggestion's budget; baseline is never omitted.

    No full 5000-iteration neural baseline is silently described as a 250-step
    run. The experimental comparator is explicitly Suggested v1, not Default.
    """
    from support.discovery_lab import starting_settings
    rule_suggestion=getattr(starting_settings,'rule_suggestion',starting_settings.suggest_settings)
    rule=rule_suggestion(method,base,context,protocol=protocol,hosted=hosted)['settings']
    entries=[('rule','Suggested v1',rule)]
    if protocol!='native' or context.get('heavy'):
        return entries
    if method=='epde':
        compact=deepcopy(rule);space=compact['search'].setdefault('search_space',{})
        space['equation_terms_max_number']=min(int(space.get('equation_terms_max_number',10)),4)
        space['deriv_fun_pow']=1
        entries.append(('compact','Compact equation search',compact))
        smooth=deepcopy(rule)
        from support.discovery_lab.starting_settings import _epde_noise
        smooth['search'].setdefault('preprocessing',{})['default_preprocessor_type']='FD'
        _epde_noise(smooth['search'],tuple(context.get('shape',())),dict(context,noise_percent=max(2.,context.get('noise_percent',0))),[])
        # On noisy inputs the rule already smooths; this policy tests a wider window.
        if settings_hash(smooth)==settings_hash(rule):
            from epde_bench.poly_options import polynomial_kwargs
            prep=smooth['search']['preprocessing'];orders=prep.get('max_deriv_order')
            if isinstance(orders,list):
                try:prep['preprocessor_kwargs']=polynomial_kwargs(13,max(orders)+1,orders,tuple(context['shape']))
                except ValueError:pass
        entries.append(('smooth','Polynomial derivative fit',smooth))
        focused=deepcopy(rule);evo=focused['search'].setdefault('evolution',{})
        work=int(evo.get('population_size',8))*int(evo.get('training_epochs',4))
        evo.update(population_size=4,training_epochs=max(1,min(8,work//4)))
        focused['search'].setdefault('search_space',{})['equation_terms_max_number']=min(8,int(rule['search'].get('search_space',{}).get('equation_terms_max_number',10)))
        entries.append(('focused','More generations, same work cap',focused))
    elif method=='pdecomp':
        original=rule.get('optimizer') or {}
        threshold=max(1e-6,float(original.get('threshold',.1)))
        for name,label,multiplier,alpha in [('sensitive','Lower sparsity threshold',.25,1e-6),
                                            ('sparse','Higher sparsity threshold',4.,1e-3),
                                            ('regularized','Stronger ridge regularization',1.,.1)]:
            s=deepcopy(rule);opt=s.setdefault('optimizer',{})
            opt.update(type='STLSQ',threshold=min(1e4,threshold*multiplier),alpha=alpha)
            entries.append((name,label,s))
    elif method=='deepmod':
        for name,label,rate,threshold in [('gentle','Lower learning rate and threshold',.001,.03),
                                         ('fast','Higher learning rate',.005,.1),
                                         ('sparse','Stronger sparsity threshold',.002,.25)]:
            s=deepcopy(rule);s.update(learning_rate=rate,threshold=threshold)
            entries.append((name,label,s))
    elif method=='discover':
        for name,label,epsilon,length,additions in [('compact','Shorter expressions',.05,9,3),
                                                   ('exploratory','Broader risk-seeking fraction',.2,15,8),
                                                   ('selective','Narrower risk-seeking fraction',.02,15,8)]:
            s=deepcopy(rule);s.update(epsilon=epsilon,max_length=length,max_add_count=additions)
            entries.append((name,label,s))
    unique=[];seen=set()
    for name,label,settings in entries:
        digest=settings_hash(settings)
        if digest in seen:continue
        try:
            if method=='epde':
                from support.discovery_lab.lab_settings import validate_public_search
                validate_public_search(settings['search'],tuple(context['shape']),override=not hosted)
                if hosted:
                    from support.discovery_lab.execution_policy import require_hosted_epde
                    require_hosted_epde(context.get('dataset'),context['shape'],context.get('variable_count',1),settings['search'])
            else:
                overrides=changes_for(method,base,settings,protocol)
                if method=='pdecomp':
                    from epde_bench.pdecomp import validate_overrides
                    validate_overrides(overrides,protocol,context.get('kind'))
                else:
                    from epde_bench.pdecomp_methods import validate_overrides
                    validate_overrides(method,overrides,protocol)
        except (ValueError,TypeError,KeyError):
            continue
        unique.append((name,label,settings));seen.add(digest)
    return unique


def changes_for(method, base, settings, protocol='native'):
    """Only supported editable fields, never external paths/framework switches."""
    if method=='epde':return {'search':deepcopy(settings['search'])}
    allowed={'pdecomp':('optimizer','boundary_trim','polynomial_degree','differentiation'),
             'deepmod':('max_iterations','max_samples','learning_rate','threshold','poly_order','diff_order'),
             'discover':('n_samples','batch_size','epsilon','max_length','max_add_count')}[method]
    return {key:deepcopy(settings[key]) for key in allowed if key in settings and settings[key]!=base.get(key)}


def semantic_signature(method, base):
    if method=='epde':
        search=base.get('search') or {};pool=search.get('search_space') or {};prep=search.get('preprocessing') or {}
        value={'tokens':pool.get('tokens'),'data_fun_pow':pool.get('data_fun_pow'),
               'orders':prep.get('max_deriv_order'),'objectives':search.get('objectives') or {}}
    elif method=='pdecomp':
        value={k:base.get(k) for k in ('targets','polynomial_degree','differentiation')}
        value['optimizer_type']=(base.get('optimizer') or {}).get('type','STLSQ')
    elif method=='deepmod':
        value={k:base.get(k) for k in ('targets','poly_order','diff_order','hidden_layers','train_fraction')}
    else:
        value={k:base.get(k) for k in ('targets','function_set','base_config')}
    return settings_hash(value)
