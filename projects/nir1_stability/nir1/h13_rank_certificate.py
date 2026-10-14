"""H13: truth-free local identifiability certificate from the SAME data.

A single traveling wave u(t,x)=f(x-c*t) satisfies u_t=-c*u_x
irrespective of whether its true governing PDE is nonlinear viscous
Burgers. On this trajectory the candidate columns (u_x, u*u_x, u_xx)
become linearly dependent. This is a mathematical ambiguity, not an
optimizer problem solved by increasing optimizer seeds.

This module is a CONTINUOUS diagnostic, not an oracle "correct" class,
not a new data budget and not a tuned precision/coverage threshold.
"""
from __future__ import annotations

import numpy as np


def normalized_spectrum(features, *, names=None):
    """Return singular-value diagnostic for L2-normalized token columns."""
    matrix=np.asarray(features,dtype=float)
    if matrix.ndim!=2 or matrix.shape[1]<2:
        raise ValueError("Expected (samples, >=2 library terms)")
    if not np.isfinite(matrix).all():
        raise ValueError("Non-finite identifiability columns")
    norm=np.linalg.norm(matrix,axis=0)
    absent=norm<=np.finfo(float).eps*max(1.,float(norm.max()))
    scaled=matrix/np.where(absent,1.,norm)
    singular=np.linalg.svd(scaled,compute_uv=False)
    ratio=(float(singular[-1]/singular[0])
           if singular.size==matrix.shape[1] and singular[0]>0
           else 0.)
    return {
        "terms":list(names) if names is not None else list(range(matrix.shape[1])),
        "n_samples":int(matrix.shape[0]),"n_terms":int(matrix.shape[1]),
        "normalized_singular_values":singular.tolist(),
        "sigma_min_over_max":ratio,
        "column_norms":norm.tolist(),
        "absent_columns":np.flatnonzero(absent).tolist(),
        "warning":"Continuous numerical diagnostic only. Threshold requires independent calibration.",
    }


def burgers_spatial_library(u,x,*,margin=8):
    """Generate generic (u_x, u*u_x, u_xx) without reference to true PDE."""
    u=np.asarray(u,dtype=float)
    x=np.asarray(x,dtype=float)
    if u.ndim!=2 or x.ndim!=1 or u.shape[1]!=len(x):
        raise ValueError("Expected u(time,space), x(space)")
    if u.shape[0]<=2*margin or u.shape[1]<=2*margin+4:
        raise ValueError("Insufficient original nodes for margin")
    if not np.all(np.diff(x)>0):
        raise ValueError("x not monotone")
    ux=np.gradient(u,x,axis=1,edge_order=2)
    uxx=np.gradient(ux,x,axis=1,edge_order=2)
    crop=(slice(margin,-margin),slice(margin,-margin))
    v=[ux[crop].ravel(),(u*ux)[crop].ravel(),uxx[crop].ravel()]
    return normalized_spectrum(np.column_stack(v),names=("u_x","u*u_x","u_xx"))


def variable_advection_library(u,x,*,margin=8):
    """Control library (u_x, x*u_x) with no truth labels."""
    u=np.asarray(u,dtype=float)
    x=np.asarray(x,dtype=float)
    if u.ndim!=2 or x.ndim!=1 or u.shape[1]!=len(x):
        raise ValueError("Expected u(time,space), x(space)")
    if not np.all(np.diff(x)>0):
        raise ValueError("x not monotone")
    ux=np.gradient(u,x,axis=1,edge_order=2)[margin:-margin,margin:-margin]
    xs=x[margin:-margin]
    return normalized_spectrum(
        np.column_stack([ux.ravel(),(ux*xs[None,:]).ravel()]),
        names=("u_x","x*u_x"))
