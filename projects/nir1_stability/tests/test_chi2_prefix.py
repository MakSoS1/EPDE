"""Opt-in E1 cumulative-score cache: strict parity and fallback tests."""

import numpy as np
import pytest

from epde.operators.common.chi2_prefix import (
    prepare_chi2_prefix, scores_from_chi2_prefix,
)
from epde.operators.common.survival import chi2_scores
from epde.operators.common.sparsity import instability_scores


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("include_intercept", [False, True])
@pytest.mark.parametrize("correlation", [0.0, 0.8, 0.995])
def test_multidimensional_chi2_path_matches_reference(weighted, include_intercept, correlation):
    rng=np.random.default_rng(20261011)
    shape=(60,100)
    n=np.prod(shape)
    p=7
    X=rng.normal(size=(n,p))
    X[:,1]=correlation*X[:,0]+np.sqrt(1-correlation**2)*X[:,1]
    y=1.1*X[:,0]-0.2*X[:,3]+0.8*X[:,6]+rng.normal(0,.03,n)
    w=rng.uniform(.5,1.5,n) if weighted else np.ones(n)
    A=np.column_stack((X,np.ones(n)))
    G=A.T@(w[:,None]*A)
    Gy=A.T@(w*y)
    cache=prepare_chi2_prefix(X,y,w,shape,G,Gy)
    assert cache is not None and len(cache["paths"])==2
    for count in (2,3,5,7):
        mask=np.zeros(p+1,dtype=bool)
        mask[:count]=True
        mask[-1]=include_intercept
        expected=instability_scores("chi2",X,y,w,shape,mask,p)
        actual=instability_scores("chi2",X,y,w,shape,mask,p,chi2_prefix=cache)
        np.testing.assert_allclose(actual,expected,atol=3e-5,rtol=1e-6)


def test_no_change_for_small_or_one_axis():
    rng=np.random.default_rng(10)
    X=rng.normal(size=(4000,3));y=rng.normal(size=4000);w=np.ones(4000)
    A=np.column_stack((X,np.ones(len(X))))
    G=A.T@A;Gy=A.T@y
    assert prepare_chi2_prefix(X,y,w,(40,100),G,Gy) is None
    X=np.tile(X[:100],(60,1));y=np.tile(y[:100],60);w=np.ones(len(y))
    A=np.column_stack((X,np.ones(len(X))))
    G=A.T@A;Gy=A.T@y
    assert prepare_chi2_prefix(X,y,w,(len(X),),G,Gy) is None
    assert prepare_chi2_prefix(X,y,w,(60,100),G,Gy,max_tensor_entries=2) is None


def test_singular_design_falls_back_to_reference():
    rng=np.random.default_rng(2026)
    X=rng.normal(size=(6000,4));X[:,1]=X[:,0]
    y=rng.normal(size=6000);w=np.ones(6000)
    A=np.column_stack((X,np.ones(len(X))))
    cache=prepare_chi2_prefix(X,y,w,(60,100),A.T@A,A.T@y)
    assert cache is not None
    assert scores_from_chi2_prefix(cache,np.ones(5,dtype=bool)) is None


def test_misaligned_mask_is_error_not_silent_reordering():
    rng=np.random.default_rng(3)
    X=rng.normal(size=(6000,3));y=rng.normal(size=6000);w=np.ones(6000)
    A=np.column_stack((X,np.ones(len(X))))
    cache=prepare_chi2_prefix(X,y,w,(60,100),A.T@A,A.T@y)
    with pytest.raises(ValueError,match="not aligned"):
        scores_from_chi2_prefix(cache,np.ones(3,dtype=bool))
