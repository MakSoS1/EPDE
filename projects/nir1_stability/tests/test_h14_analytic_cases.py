"""Frozen H14 holdout analytical PDE residuals before any search outputs."""
import numpy as np
import pytest

from projects.pic.epde_bench.datasets import load

CASES=(
"nir1_h14_damped_wave",
"nir1_h14_cole_hopf",
"nir1_h14_kpp_front",
"nir1_h14_linear_ks",
"nir1_h14_beam",
"nir1_h14_airy_transport",
"nir1_h14_cubic_saturation",
"nir1_h14_driven_advection",
"nir1_h14_variable_diffusivity",
"nir1_h14_liouville_growth",
"nir1_h14_hyperdiffusion_growth",
)


@pytest.mark.parametrize("dataset",CASES)
def test_analytic_solution_obeys_stated_governing_law(dataset):
    problem=load(dataset)
    t,x=problem.grids
    u=problem.data["u"]
    assert u.shape==t.shape==x.shape==(81,96)
    assert np.isfinite(u).all()
    assert len(problem.truth_systems)==1
    dt=float(t[1,0]-t[0,0])
    dx=float(x[0,1]-x[0,0])
    def diff(data,h,axis,order):
        for _ in range(order):
            data=np.gradient(data,h,axis=axis,edge_order=2)
        return data
    ut=diff(u,dt,0,1)
    utt=diff(u,dt,0,2)
    ux=diff(u,dx,1,1)
    uxx=diff(u,dx,1,2)
    uxxx=diff(u,dx,1,3)
    uxxxx=diff(u,dx,1,4)
    kind=dataset.replace("nir1_h14_","")
    if kind=="damped_wave":
        residual=utt-.4225*uxx+.22*ut
    elif kind=="cole_hopf":
        residual=ut+u*ux-.17*uxx
    elif kind=="kpp_front":
        residual=ut-.2*uxx-.5*u+.5*u*u
    elif kind=="linear_ks":
        residual=ut+.08*uxx+.01*uxxxx
    elif kind=="beam":
        residual=utt+.03*uxxxx
    elif kind=="airy_transport":
        residual=ut+.2*ux+.03*uxxx
    elif kind=="cubic_saturation":
        residual=ut-.55*u+.55*u**3
    elif kind=="driven_advection":
        residual=ut+(.3+.2*np.sin(2*t))*ux
    elif kind=="variable_diffusivity":
        residual=ut-.07*(1+x*x)*uxx-.14*x*ux
    elif kind=="liouville_growth":
        residual=ut-.24*u+.24*x*ux
    elif kind=="hyperdiffusion_growth":
        residual=ut-.2*u+.012*uxxxx
    else:
        raise AssertionError("Unvalidated physics law")
    crop=(slice(8,-8),slice(8,-8))
    rhs=(utt if kind in ("damped_wave","beam") else ut)[crop]
    relative=float(np.linalg.norm(residual[crop])/
                   max(np.linalg.norm(rhs),1e-9))
    assert relative < .22, (kind,relative)
