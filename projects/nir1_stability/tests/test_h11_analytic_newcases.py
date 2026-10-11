"""Analytic validation for H11 frozen extra-equation physical holdout.

These checks validate SIMULATOR PHYSICS, not the search. It is important
not to give the true derivative terms or coefficients to the estimator:
benchmark truth is read only for offline scoring after search.fit().
"""
import numpy as np
import pytest

from projects.pic.epde_bench.datasets import load

CASES=(
 "nir1_h11_gaussian_heat","nir1_h11_advection_decay",
 "nir1_h11_viscous_burgers","nir1_h11_klein_gordon",
 "nir1_h11_biharmonic","nir1_h11_airy","nir1_h11_shear",
 "nir1_h11_logistic_field","nir1_h11_gauss_reactdiff",
)


@pytest.mark.parametrize("name",CASES)
def test_analytic_family_obeys_predeclared_physics(name):
    case=load(name)
    assert case.truth_systems and len(case.truth_systems)==1
    assert len(case.grids)==2 and case.dim==1
    t,x=case.grids
    u=case.data["u"]
    assert t.shape==x.shape==u.shape==(81,96)
    assert np.isfinite(u).all()
    dt=float(t[1,0]-t[0,0])
    dx=float(x[0,1]-x[0,0])
    def derivative(a,h,axis,n):
        for _ in range(n):
            a=np.gradient(a,h,axis=axis,edge_order=2)
        return a
    ut=derivative(u,dt,0,1)
    utt=derivative(u,dt,0,2)
    ux=derivative(u,dx,1,1)
    uxx=derivative(u,dx,1,2)
    uxxx=derivative(u,dx,1,3)
    uxxxx=derivative(u,dx,1,4)
    if name.endswith("gaussian_heat"):
        residual=ut-.12*uxx
    elif name.endswith("advection_decay"):
        residual=ut+.57*ux+.24*u
    elif name.endswith("viscous_burgers"):
        residual=ut+u*ux-.08*uxx
    elif name.endswith("klein_gordon"):
        residual=utt-.49*uxx+.1225*u
    elif name.endswith("biharmonic"):
        residual=ut+.018*uxxxx
    elif name.endswith("airy"):
        residual=ut+.047*uxxx
    elif name.endswith("shear"):
        residual=ut+x*ux
    elif name.endswith("logistic_field"):
        residual=ut-.8*u+.8*u*u
    elif name.endswith("gauss_reactdiff"):
        residual=ut-.13*uxx-.27*u
    else:
        raise AssertionError("No physics verification for case")
    resid=residual[8:-8,8:-8]
    target=(utt if name.endswith("klein_gordon") else ut)[8:-8,8:-8]
    relative=np.linalg.norm(resid)/max(np.linalg.norm(target),1e-10)
    assert relative < .20, f"{name}: PDE error too high {relative}"


def test_holdout_bank_contains_higher_order_truths_to_falsify_mdl():
    high=[load("nir1_h11_biharmonic").truth_systems,
          load("nir1_h11_airy").truth_systems]
    strings=repr(high)
    assert "d^4u/dx1^4" in strings
    assert "d^3u/dx1^3" in strings


def test_h11_selector_ignores_truth_and_chooses_smallest_fitness():
    from projects.nir1_stability.nir1.causal_mdl_selector import select_min_causal_description
    obj=[[.18,0.],[.11,.8],[.2,0.]]
    idx=select_min_causal_description(obj)
    assert idx==1
    assert idx==select_min_causal_description(obj,truth_labels=[True,False,False])
    assert idx==select_min_causal_description(obj,truth_labels=[False,True,False])
    assert select_min_causal_description([[float("nan"),0.],[.01,0.]])==1
    assert select_min_causal_description([]) is None
