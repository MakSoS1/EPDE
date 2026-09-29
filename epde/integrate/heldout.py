"""The held-out time split shared by every solver backend.

Moved verbatim out of ``deepxde_integration`` (which re-exports all four
names), so a backend that does not use DeepXDE -- ``basis_integration`` --
never imports it. ``import deepxde`` changes torch's process-wide default
device, which is reason enough to keep it out of a path that does not need it.

``time_split`` is the ONE rule for what the observation term sees and where
the fitness host scores; every backend must use it, not a copy.
"""
import math
import warnings
from typing import NamedTuple, Tuple

import numpy as np

from epde.globals import EPDEUsageWarning


class DeepXDEConfigError(ValueError):
    """A ``deepxde_config`` that cannot describe a run.

    Raised past the solve's blanket ``except Exception``: a misconfigured split
    must stop the search, not turn into a NaN loss for every candidate alike.
    """


class TimeSplit(NamedTuple):
    """The three time blocks, and where their boundaries fell.

    ``test`` is INVARIANT to ``val_frac`` -- the validation block is carved out
    of the fit window, never out of the scored tail. That keeps every number
    measured against the old two-way split directly comparable, and it keeps the
    scored block sacred: selecting an iterate on a slice adjacent to the one you
    report would inflate the separation for reasons unrelated to the equation.
    """
    train: np.ndarray       #: observation term, and the variance yardsticks
    val: np.ndarray         #: iterate selection only; empty iff val_frac == 0
    test: np.ndarray        #: where the host scores
    t_train: float          #: last training level
    t_fit: float            #: last non-test level (the old ``t_split``, always)
    n_levels: Tuple[int, int, int]      #: (n_train, n_val, n_test)


def time_split(t_inner, train_frac: float, val_frac: float = 0.0) -> TimeSplit:
    """Partition inner-domain points by time into train / val / test.

    ``n_fit = ceil(train_frac * n_levels)`` sets the test boundary exactly as it
    always has; ``val`` is then the trailing ``ceil(val_frac * n_levels)`` slice
    of that fit window, so raising ``val_frac`` spends observation levels and
    never touches the scored tail. One rule, shared by the adapter (what the
    observation term sees) and the fitness host (where the solution is scored).

    ``val_frac=0`` reproduces the two-way split bit-for-bit by construction:
    ``n_val = 0`` makes ``train`` literally the old mask and ``test`` its
    complement.

    The validation block is the strip just past supervision, where only the
    candidate equation drives the solution -- near-extrapolation of the same
    kind the tail measures, rather than an interpolation check.
    """
    if not 0.0 < float(train_frac) < 1.0:
        raise ValueError(f'train_frac must lie strictly between 0 and 1, '
                         f'got {train_frac!r}')
    if not 0.0 <= float(val_frac) < 1.0:
        raise DeepXDEConfigError(f'val_frac must lie in [0, 1), got {val_frac!r}')
    t = np.asarray(t_inner).reshape(-1)
    levels = np.unique(t)
    n = len(levels)
    n_fit = int(math.ceil(float(train_frac) * n))
    if n_fit >= n:
        raise ValueError(
            f'train_frac={train_frac} leaves no held-out time level '
            f'({n} levels on the inner domain)')
    n_val = int(math.ceil(float(val_frac) * n)) if float(val_frac) > 0.0 else 0
    n_train = n_fit - n_val
    if n_train < 1:
        raise DeepXDEConfigError(
            f'train_frac={train_frac}, val_frac={val_frac} leave {n_train} training '
            f'time level(s) of {n} (fit {n_fit}, val {n_val}, test {n - n_fit}); '
            f'lower val_frac or use a finer time grid')
    if 0 < n_val < 3:
        warnings.warn(
            f'val_frac={val_frac} carves only {n_val} time level(s) of {n}: a val '
            f'block shorter than 3 levels measures continuity at the training '
            f'boundary, not extrapolation', EPDEUsageWarning, stacklevel=2)
    t_train = float(levels[n_train - 1])
    t_fit = float(levels[n_fit - 1])
    return TimeSplit(t <= t_train, (t > t_train) & (t <= t_fit), t > t_fit,
                     t_train, t_fit, (n_train, n_val, n - n_fit))


def variance_weight(values) -> float:
    """``1 / Var(values)`` (population variance), the yardstick that makes a
    mean-squared loss term dimensionless.

    Floored at the dtype's smallest positive value rather than by an additive
    epsilon: an additive constant is a scale, and it breaks the invariance
    under rescaling the field that justifies weight 1. The floor only matters
    for a constant channel.
    """
    var = float(np.var(np.asarray(values, dtype=np.float64)))
    return 1.0 / max(var, float(np.finfo(np.float64).tiny))
