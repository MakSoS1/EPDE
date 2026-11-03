"""Bounded, numerical-only descriptors. Names and reference laws never enter."""
from __future__ import annotations
import math
import numpy as np

VERSION = 'numeric-descriptors-v1'
FEATURE_NAMES = ('axes', 'variables', 'log_cells', 'log_time_points', 'log_space_points',
                 'log_time_span', 'log_space_span', 'uniform_fraction', 'log_amplitude',
                 'relative_offset', 'skew', 'kurtosis', 'time_roughness', 'space_roughness',
                 'time_curvature', 'space_curvature', 'spectral_entropy', 'high_frequency')


def _spectrum(a, axis):
    a = a - np.mean(a, axis=axis, keepdims=True)
    power = np.abs(np.fft.rfft(a, axis=axis))**2
    power = np.mean(power, axis=tuple(i for i in range(a.ndim) if i != axis))
    power = np.asarray(power).reshape(-1)[1:]
    total = float(np.sum(power))
    if not total > 1e-24 or len(power) < 2:
        return 0., 0.
    weights = power/total
    entropy = -float(np.sum(weights*np.log(weights+1e-30)))/math.log(len(weights))
    return entropy, float(np.sum(weights[len(weights)//2:]))


def describe_arrays(arrays):
    """Inspect at most 64 points/axis and four variables; reject invalid values.

    No learned noise estimator is implied by roughness: genuine oscillations
    and measurement noise can both increase these descriptors.
    """
    axis_keys = sorted((k for k in arrays if k.startswith('axis_') and k != 'axis_names'),
                       key=lambda k: int(k[5:]))
    fields = [np.asarray(v) for k,v in arrays.items() if k.startswith('var_')]
    if not 1 <= len(axis_keys) <= 2 or not 1 <= len(fields) <= 4:
        raise ValueError('Advisor descriptors support 1–2 axes and 1–4 variables')
    axes = [np.asarray(arrays[k], dtype=float).reshape(-1) for k in axis_keys]
    shape = tuple(len(a) for a in axes)
    if any(n < 8 for n in shape) or math.prod(shape) > 300_000:
        raise ValueError('Input is outside the bounded descriptor size')
    uniform = []
    spans = []
    for a in axes:
        if not np.isfinite(a).all() or not np.all(np.diff(a)>0):
            raise ValueError('Coordinates must be finite and strictly increasing')
        uniform.append(float(np.allclose(np.diff(a),np.mean(np.diff(a)),rtol=1e-4,atol=1e-9)))
        spans.append(float(a[-1]-a[0]))
    values=[]
    for field in fields:
        if field.shape != shape or not np.isfinite(field).all():
            raise ValueError('Fields must have matching shape and finite values')
        a = np.asarray(field, dtype=float)
        for axis,n in enumerate(shape):
            a=np.take(a,np.linspace(0,n-1,min(n,64),dtype=int),axis=axis)
        amplitude = float(np.max(np.abs(a)))
        # Scale before subtracting/squaring to avoid overflow on valid huge values.
        scaled = a/(amplitude or 1.)
        std = float(np.std(scaled)); mean=float(np.mean(scaled))
        z=(scaled-mean)/max(std,1e-12)
        rough=[float(np.sqrt(np.mean(np.diff(z,axis=i)**2))) for i in range(a.ndim)]
        curvature=[float(np.sqrt(np.mean(np.diff(z,n=2,axis=i)**2))) for i in range(a.ndim)]
        entropy, high = _spectrum(z, a.ndim-1)
        values.append([math.log1p(amplitude),float(np.clip(mean/max(std,1e-12),-100,100)),
                       float(np.mean(z**3)),float(np.mean(z**4)),rough[0],rough[-1] if a.ndim>1 else 0.,
                       curvature[0],curvature[-1] if a.ndim>1 else 0.,entropy,high])
    row=[len(shape),len(fields),math.log1p(math.prod(shape)),math.log1p(shape[0]),
         math.log1p(shape[1]) if len(shape)>1 else 0.,math.log1p(spans[0]),
         math.log1p(spans[1]) if len(shape)>1 else 0.,float(np.mean(uniform)),
         *np.mean(values,axis=0).tolist()]
    if not np.isfinite(row).all():
        raise ValueError('Nonfinite numerical descriptors')
    return dict(zip(FEATURE_NAMES,map(float,row)))


def describe_problem(problem):
    arrays={f'var_{k}':v for k,v in problem.data.items()}
    for axis, grid in enumerate(problem.grids):
        g=np.asarray(grid)
        if g.ndim>1:
            index=[0]*g.ndim;index[axis]=slice(None);g=g[tuple(index)]
        arrays[f'axis_{axis}']=g
    return describe_arrays(arrays)
