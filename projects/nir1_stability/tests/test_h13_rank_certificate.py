"""H13 rank diagnostic needs no ground-truth coefficients or extra data."""
import numpy as np
import pytest

from projects.nir1_stability.nir1.h13_rank_certificate import (
    normalized_spectrum, burgers_spatial_library, variable_advection_library,
)


def test_column_scale_invariant_and_dependent_columns():
    rng=np.random.default_rng(40)
    a=rng.normal(size=900)
    b=rng.normal(size=900)
    X=np.column_stack([a,b,3*a-b])
    out=normalized_spectrum(X,names=["a","b","3a-b"])
    out2=normalized_spectrum(X*np.array([10,.001,3]))
    assert out["sigma_min_over_max"]<1e-12
    assert out2["sigma_min_over_max"]<1e-12
    assert out["terms"]==["a","b","3a-b"]


def test_same_grid_burgers_wave_vs_nonconstant_shear():
    t=np.linspace(0,1.8,81)[:,None]
    x=np.linspace(-np.pi,np.pi,96,endpoint=False)
    xx=x[None,:]
    speed,amplitude,nu=.6,.5,.08
    u=speed-amplitude*np.tanh(amplitude/(2*nu)*(xx-speed*t))
    burger=burgers_spatial_library(u,x)
    # Identifiability loss: normalized ux, uux, uxx are nearly dependent
    # even though the measured field contains many points.
    assert burger["sigma_min_over_max"]<.02
    q=xx*np.exp(-t)
    shear=np.sin(1.6*q+.1)+.25*np.cos(2.3*q-.15)
    control=variable_advection_library(shear,x)
    assert control["sigma_min_over_max"]>.4
    assert burger["n_samples"]==control["n_samples"]
    assert burger["sigma_min_over_max"]<control["sigma_min_over_max"]/20


def test_invalid_and_nonfinite_data_rejected():
    with pytest.raises(ValueError):
        normalized_spectrum(np.asarray([1,2,3]))
    with pytest.raises(ValueError):
        normalized_spectrum([[1,float("nan")],[2,3]])
