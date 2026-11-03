"""Small, checked evaluator for CatBoost's numeric oblivious-tree export.

Only trusted packaged models are loaded. Training requires CatBoost; prediction
uses NumPy and is parity-tested against the native CatBoost implementation.
"""
from __future__ import annotations
import json
import numpy as np


def export_numeric(model, path):
    model.save_model(str(path),format='json')
    original=json.loads(path.read_text())
    info=original['features_info']
    if info.get('categorical_features') or info.get('ctrs'):
        raise ValueError('Only numeric CatBoost models may be exported')
    mapping={f['feature_index']:f['flat_feature_index'] for f in info['float_features']}
    scale,bias=original.get('scale_and_bias',[1.,[0.]])
    if len(bias)!=1:
        raise ValueError('Only single-target models are supported')
    trees=[]
    for tree in original['oblivious_trees']:
        splits=[]
        for split in tree.get('splits') or []:
            if split['split_type']!='FloatFeature':
                raise ValueError('Unsupported CatBoost split')
            splits.append([mapping[split['float_feature_index']],split['border']])
        trees.append({'splits':splits,'leaves':tree['leaf_values']})
    out={'schema':'numeric-catboost/1','feature_count':len(model.feature_names_),
         'scale':float(scale),'bias':float(bias[0]),'trees':trees}
    # Fail before a model is published, not on the first visitor's request.
    predict(out,np.zeros((1,out['feature_count'])))
    return out


def predict(model, features):
    x=np.asarray(features,dtype=float)
    count=model.get('feature_count')
    if (model.get('schema')!='numeric-catboost/1' or type(count) is not int
            or not 1<=count<=128 or x.ndim!=2 or x.shape[1]!=count or not np.isfinite(x).all()):
        raise ValueError('Invalid numeric CatBoost model or feature matrix')
    # Native CatBoost quantizes float inputs as float32 before testing borders.
    # Comparing float64 neighbours directly can choose a different leaf.
    with np.errstate(over='ignore'):
        x=x.astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError('Features exceed CatBoost float32 range')
    trees=model.get('trees')
    if not isinstance(trees,list) or not 1<=len(trees)<=1000:
        raise ValueError('Invalid CatBoost tree count')
    result=np.zeros(len(x))
    for tree in trees:
        splits=tree['splits'];leaves=np.asarray(tree['leaves'],dtype=float)
        if len(splits)>10 or leaves.shape!=(2**len(splits),) or not np.isfinite(leaves).all():
            raise ValueError('Invalid CatBoost leaves')
        indexes=np.zeros(len(x),dtype=np.int64)
        for bit,(feature,border) in enumerate(splits):
            if type(feature) is not int or not 0<=feature<count or not np.isfinite(border):
                raise ValueError('Invalid CatBoost split')
            indexes |= (x[:,feature]>border).astype(np.int64)<<bit
        result+=leaves[indexes]
    scale,bias=float(model['scale']),float(model['bias'])
    if not np.isfinite([scale,bias]).all():
        raise ValueError('Invalid CatBoost scaling')
    return result*scale+bias
