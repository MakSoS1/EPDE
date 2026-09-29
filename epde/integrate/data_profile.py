"""What the observed field says about a solve, read from the TRAIN window only.

A solver that has to extrapolate into a held-out tail must not be configured
from that tail. Everything here is computed from the time levels up to
``t_train`` (the last level the observation term sees). Replacing every later
level with NaN leaves the profile bit-identical, which the tests pin.

* **axes**: bounds, size and spacing, giving the affine map onto [-1, 1] that a
  network input or a Chebyshev basis wants.
* **periodicity** per SPATIAL axis (time never). This is a statement about the
  data, since periodic boundary conditions are not declared anywhere in EPDE.
  At each train level it compares the jump across the seam and the second
  difference across it with the largest interior jump and second difference.
  A smooth periodic field continues across the seam no worse than it varies
  inside. The verdict is the MEDIAN over train levels of the larger ratio,
  compared with 1. A single level is not enough: Allen-Cahn's initial
  condition x^2 cos(pi x) is not smoothly periodic (seam ratio 37.9 at t=0)
  and only becomes so from t ~ 0.24 on. Both grid conventions are tried: the
  last point one spacing short of the period ('excluded', e.g. AC and Burgers)
  or the last point repeating the first ('included').
* **spectral size** per axis: the number of modes that hold ``energy`` of the
  train-window FLUCTUATION (the mean is left out, so adding a constant to the
  field changes nothing). Periodic axes use an FFT, anything else a Chebyshev
  least-squares fit on the axis's own interval. For
  time, the count is scaled by full extent / train extent, because the basis
  must also cover the tail.
* **variables**: train-window mean and standard deviation.

numpy/scipy only, and no top-level ``epde`` import, so standalone scripts can
load this file by path. The caller supplies ``t_train`` (e.g.
``time_split(...).t_train``); nothing here decides the split.
"""
import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

#: Share of the train-window energy the spectral counts must hold.
DEFAULT_ENERGY = 0.99999


@dataclass(frozen=True)
class AxisProfile:
    axis: int                    #: EPDE axis index (0 = time)
    lo: float
    hi: float
    n: int
    spacing: float               #: median spacing
    uniform: bool
    periodic: bool
    period: Optional[float]      #: None unless periodic
    endpoint: Optional[str]      #: 'excluded' | 'included' | None
    seam_ratio: Optional[float]  #: median seam statistic ratio (<= 1: periodic); None for time
    modes: int                   #: spectral size the axis needs (see module doc)

    def to_unit(self, values):
        """Affine map of this axis onto [-1, 1]."""
        return 2.0 * (np.asarray(values, dtype=np.float64) - self.lo) / (self.hi - self.lo) - 1.0

    @property
    def affine(self) -> Tuple[float, float]:
        """``(scale, shift)`` with ``unit = scale * x + shift``."""
        scale = 2.0 / (self.hi - self.lo)
        return scale, -1.0 - scale * self.lo


@dataclass(frozen=True)
class VariableProfile:
    name: str
    mean: float
    std: float


@dataclass(frozen=True)
class DataProfile:
    axes: Tuple[AxisProfile, ...]
    variables: Tuple[VariableProfile, ...]
    t_train: float
    n_train_levels: int
    n_levels: int
    energy: float

    @property
    def periodic_axes(self) -> Tuple[int, ...]:
        return tuple(a.axis for a in self.axes if a.periodic)

    def variable(self, name: str) -> VariableProfile:
        for v in self.variables:
            if v.name == name:
                return v
        raise KeyError(name)


# ------------------------------------------------------------ helpers
def axis_coordinates(grid: np.ndarray, axis: int) -> np.ndarray:
    """The 1-D coordinate vector of ``axis`` from an ij-indexed meshgrid array."""
    g = np.asarray(grid, dtype=np.float64)
    if g.ndim == 1:
        return g
    return np.moveaxis(g, axis, 0)[(slice(None),) + (0,) * (g.ndim - 1)]


def train_level_mask(t_coords: np.ndarray, t_train: float) -> np.ndarray:
    t = np.asarray(t_coords, dtype=np.float64)
    mask = t <= float(t_train)
    if not mask.any():
        raise ValueError(f"t_train={t_train} precedes every time level ({t.min()}..{t.max()})")
    return mask


def _ratio(num: float, den: float) -> float:
    if den > 0.0:
        return num / den
    return 0.0 if num == 0.0 else math.inf


def _seam_ratio(lines: np.ndarray, endpoint: str) -> float:
    """Seam statistic of one time level. ``lines`` is (n_lines, n) along the axis."""
    u = lines
    d1 = np.abs(np.diff(u, axis=1))
    d2 = np.abs(u[:, 2:] - 2.0 * u[:, 1:-1] + u[:, :-2])
    m1 = float(d1.max()) if d1.size else 0.0
    m2 = float(d2.max()) if d2.size else 0.0
    jump = float(np.abs(u[:, 0] - u[:, -1]).max())
    if endpoint == "excluded":
        # the point after u[-1] is u[0]
        s2 = float(np.maximum(np.abs(u[:, 1] - 2.0 * u[:, 0] + u[:, -1]),
                              np.abs(u[:, 0] - 2.0 * u[:, -1] + u[:, -2])).max())
    else:
        # u[-1] IS u[0]; the point after it is u[1]
        s2 = float(np.abs(u[:, 1] - 2.0 * u[:, 0] + u[:, -2]).max())
    return max(_ratio(jump, m1), _ratio(s2, m2))


def _endpoint_convention(windows: Dict[str, np.ndarray], axis: int) -> str:
    """Which grid convention a field that passes BOTH seam tests uses.

    Under 'included' the first and last points are the same point, so their
    difference is ~0 next to the differences beside the seam. Under 'excluded'
    they are neighbours, so it is of the same size. A field flat at the seam
    (Burgers decays to ~0 at both ends) cannot tell the two apart. Then the
    answer is 'excluded', the convention of FFT grids and of every periodic
    dataset in the repo.
    """
    votes = []
    for w in windows.values():
        for li in range(w.shape[0]):
            u = _lines(w[li], axis - 1)
            m1 = float(np.abs(np.diff(u, axis=1)).max())
            beside = float(np.maximum(np.abs(u[:, 1] - u[:, 0]),
                                      np.abs(u[:, -1] - u[:, -2])).max())
            if beside <= 1e-8 * m1:
                continue                          # flat seam: no information
            votes.append(float(np.abs(u[:, 0] - u[:, -1]).max()) / beside)
    if not votes:
        return "excluded"
    return "included" if float(np.median(votes)) < 0.5 else "excluded"


def _lines(slab: np.ndarray, axis: int) -> np.ndarray:
    """All 1-D lines of ``slab`` along ``axis`` as rows."""
    moved = np.moveaxis(slab, axis, -1)
    return moved.reshape(-1, moved.shape[-1])


def _count_modes(power: np.ndarray, energy: float) -> int:
    """Smallest K with sum(power[:, 1:K+1]) >= energy * sum(power[:, 1:]) per
    row, max over rows. Column 0 (the mean) is left out of both sides, so the
    count is invariant to adding a constant to the field. A row with no
    fluctuation needs K = 0."""
    fluct = power[:, 1:]
    total = fluct.sum(axis=1, keepdims=True)
    live = total[:, 0] > 0.0
    if not live.any() or fluct.shape[1] == 0:
        return 0
    frac = np.cumsum(fluct[live], axis=1) / total[live]
    return int((np.argmax(frac >= energy - 1e-15, axis=1) + 1).max())


def _fourier_count(lines: np.ndarray, endpoint: str, energy: float) -> int:
    """Largest wavenumber K the periodic lines need (basis size 2K+1)."""
    u = lines if endpoint == "excluded" else lines[:, :-1]
    n = u.shape[1]
    spec = np.abs(np.fft.rfft(u, axis=1)) ** 2
    weights = np.full(spec.shape[1], 2.0)
    weights[0] = 1.0
    if n % 2 == 0:
        weights[-1] = 1.0
    return _count_modes(spec * weights, energy)


def _chebyshev_count(coords: np.ndarray, lines: np.ndarray, energy: float) -> int:
    """Chebyshev coefficients the lines need on their own interval.

    The smallest m such that the least-squares fit by T_0..T_{m-1} leaves at
    most ``1 - energy`` of the lines' fluctuation energy. A DCT would read the
    window as one half of a mirrored period, and the kink at its ends makes
    the spectrum decay algebraically (measured: AC needed 46-51 of 51 levels).
    Chebyshev polynomials have no such seam, and they are what the basis
    solver uses on these axes. One QR of the Vandermonde matrix gives every
    degree at once: the residual after m columns is the total minus the
    squared projections onto the first m orthonormal directions.
    """
    from numpy.polynomial import chebyshev as C
    c = np.asarray(coords, dtype=np.float64)
    n = len(c)
    if n < 2:
        return 1
    xm = 2.0 * (c - c.min()) / (c.max() - c.min()) - 1.0
    y = lines.T                                             # (n, n_lines)
    fluct = float(((y - y.mean(axis=0, keepdims=True)) ** 2).sum())
    if fluct == 0.0:
        return 1
    q, _ = np.linalg.qr(C.chebvander(xm, n - 1))
    proj = ((q.T @ y) ** 2).sum(axis=1)                      # energy per direction
    total = float((y ** 2).sum())
    left = total - np.cumsum(proj)
    ok = np.nonzero(left <= (1.0 - energy) * fluct)[0]
    return int(ok[0] + 1) if len(ok) else n


# ------------------------------------------------------------ profile
def build_profile(full_grids: Sequence[np.ndarray], raw_fields: Dict[str, np.ndarray],
                  t_train: float, energy: float = DEFAULT_ENERGY,
                  seam_threshold: float = 1.0) -> DataProfile:
    """Profile of the observed fields on their FULL grid, train window only.

    ``full_grids`` are ij-indexed coordinate arrays (axis 0 = time) or, for an
    ODE, the single time vector. ``raw_fields`` maps variable name to the
    observed field on that grid, as handed to ``createTrajectory``.
    """
    grids = [np.asarray(g, dtype=np.float64) for g in full_grids]
    ndim = len(grids)
    coords = [axis_coordinates(grids[a], a) for a in range(ndim)]
    level = train_level_mask(coords[0], t_train)
    n_train = int(level.sum())
    fields = {k: np.asarray(v, dtype=np.float64) for k, v in raw_fields.items()}
    for name, f in fields.items():
        if f.shape != tuple(len(c) for c in coords):
            raise ValueError(f"field {name!r} has shape {f.shape}, grid is "
                             f"{tuple(len(c) for c in coords)}")
    windows = {k: f[level] for k, f in fields.items()}
    for name, w in windows.items():
        if not np.isfinite(w).all():
            raise ValueError(f"field {name!r} is not finite inside the train window")

    axes = []
    for a in range(ndim):
        c = coords[a]
        lo, hi, n = float(c.min()), float(c.max()), len(c)
        steps = np.diff(c)
        spacing = float(np.median(steps)) if len(steps) else 0.0
        uniform = bool(len(steps) == 0 or np.allclose(steps, spacing, rtol=1e-6, atol=0.0))
        periodic, period, endpoint, seam = False, None, None, None
        if a == 0:
            # time: Chebyshev count over the train window, scaled to the full extent
            counts = [_chebyshev_count(c[level], _lines(w, 0), energy)
                      for w in windows.values()]
            m_train = max(counts) if counts else 0
            t_lo, t_tr = float(c[level].min()), float(c[level].max())
            scale = (hi - lo) / (t_tr - t_lo) if t_tr > t_lo else 1.0
            modes = int(min(n, math.ceil(m_train * scale)))
        else:
            ratios = {}
            if uniform and n >= 4:
                for conv in ("excluded", "included"):
                    per_level = []
                    for w in windows.values():
                        for li in range(w.shape[0]):
                            per_level.append(_seam_ratio(_lines(w[li], a - 1), conv))
                    ratios[conv] = float(np.median(per_level))
            if ratios:
                passing = [k for k in ratios if ratios[k] <= seam_threshold]
                if len(passing) == 2:
                    endpoint = _endpoint_convention(windows, a)
                else:
                    endpoint = min(ratios, key=lambda k: (ratios[k], k != "excluded"))
                seam = ratios[endpoint]
                periodic = seam <= seam_threshold
            if periodic:
                period = spacing * (n if endpoint == "excluded" else n - 1)
                counts = [_fourier_count(_lines(w, a), endpoint, energy)
                          for w in windows.values()]
            else:
                endpoint = None if not ratios else endpoint
                counts = [_chebyshev_count(c, _lines(w, a), energy)
                          for w in windows.values()]
            modes = max(counts) if counts else 0
        axes.append(AxisProfile(axis=a, lo=lo, hi=hi, n=n, spacing=spacing, uniform=uniform,
                                periodic=periodic, period=period,
                                endpoint=endpoint if periodic else None,
                                seam_ratio=seam, modes=int(modes)))

    variables = tuple(VariableProfile(k, float(np.mean(w)), float(np.std(w)))
                      for k, w in windows.items())
    return DataProfile(axes=tuple(axes), variables=variables, t_train=float(t_train),
                       n_train_levels=n_train, n_levels=len(coords[0]), energy=float(energy))


def gradient_density(full_grids: Sequence[np.ndarray], field: np.ndarray, t_train: float,
                     kind: str = "all") -> np.ndarray:
    """``1 + |grad u| / mean|grad u|`` on the full grid, from the train window.

    ``kind='space'`` differentiates the spatial axes only; ``'all'`` includes
    time. Levels past ``t_train`` copy the last train level, since the tail
    must not be read. A constant field gives density 1 everywhere.
    """
    if kind not in ("all", "space"):
        raise ValueError(f"kind must be 'all' or 'space', got {kind!r}")
    grids = [np.asarray(g, dtype=np.float64) for g in full_grids]
    ndim = len(grids)
    coords = [axis_coordinates(grids[a], a) for a in range(ndim)]
    level = train_level_mask(coords[0], t_train)
    u = np.asarray(field, dtype=np.float64)[level]
    if np.ptp(u) == 0.0:
        # np.gradient's non-uniform stencil leaves ~1e-16 on a constant field,
        # which the normalisation below would blow up into structure
        return np.ones(np.asarray(field).shape, dtype=np.float64)
    axes = range(ndim) if kind == "all" else range(1, ndim)
    sq = np.zeros_like(u)
    for a in axes:
        c = coords[a][level] if a == 0 else coords[a]
        if len(c) < 2:
            continue
        sq += np.gradient(u, c, axis=a) ** 2
    mag = np.sqrt(sq)
    mean = float(mag.mean())
    dens_train = 1.0 + (mag / mean if mean > 0.0 else 0.0 * mag)
    out = np.empty(np.asarray(field).shape, dtype=np.float64)
    out[level] = dens_train
    if (~level).any():
        out[~level] = dens_train[-1]
    return out
