"""Validate immutable new analytic ODE/PDE benchmark BEFORE optimization.

Not a model-fit test: numerical finite differences of generated exact solutions
are compared to their governing equations, independent of EPDE selection.
"""

import numpy as np
import pytest

from projects.pic.epde_bench.datasets import load
from projects.pic.epde_bench.metrics import canonical_tokens


ODES = ["nir1_decay", "nir1_logistic", "nir1_quadratic",
        "nir1_cubic", "nir1_harmonic"]
PDES = ["nir1_advection", "nir1_heat", "nir1_reactiondiff",
        "nir1_wave", "nir1_advdiff"]


def _error(residual, scales):
    denom = max(float(np.sqrt(np.mean(np.asarray(scales)**2))), 1e-6)
    return float(np.sqrt(np.mean(np.asarray(residual)**2))/denom)


@pytest.mark.parametrize("system", ODES + PDES)
def test_analytic_benchmark_produces_finite_reproducible_data(system):
    first, second = load(system), load(system)
    assert first.truth and len(first.truth) == 1
    assert len(canonical_tokens(first.truth)) == 1
    assert first.source == "synthetic"
    assert first.name == system
    u = first.data["u"]
    assert np.issubdtype(u.dtype, np.floating)
    assert np.all(np.isfinite(u))
    assert np.array_equal(u, second.data["u"])
    assert len(first.grids) == u.ndim
    assert all(np.asarray(g).shape == u.shape for g in first.grids)
    assert len(first.truth_systems) == 1


@pytest.mark.parametrize("system", ODES)
def test_ode_gov_equation_finite_difference_residual(system):
    pr = load(system)
    t = pr.grids[0]
    u = pr.data["u"]
    ut = np.gradient(u, t, edge_order=2)
    utt = np.gradient(ut, t, edge_order=2)
    if system == "nir1_decay":
        residual, scale = ut + .7*u, ut
    elif system == "nir1_logistic":
        residual, scale = ut - .65*u + .325*u*u, ut
    elif system == "nir1_quadratic":
        residual, scale = ut + .22*u*u, ut
    elif system == "nir1_cubic":
        residual, scale = ut -u +u**3, ut
    else:
        residual, scale = utt + 1.69*u, utt
    assert _error(residual[4:-4], scale[4:-4]) < .01


@pytest.mark.parametrize("system", PDES)
def test_pde_gov_equation_finite_difference_residual(system):
    pr = load(system)
    t = pr.grids[0][:, 0]
    x = pr.grids[1][0, :]
    u = pr.data["u"]
    ut = np.gradient(u, t, axis=0, edge_order=2)
    utt = np.gradient(ut, t, axis=0, edge_order=2)
    ux = np.gradient(u, x, axis=1, edge_order=2)
    uxx = np.gradient(ux, x, axis=1, edge_order=2)
    if system == "nir1_advection":
        residual, scale = ut + .7*ux, ut
    elif system == "nir1_heat":
        residual, scale = ut - .17*uxx, ut
    elif system == "nir1_reactiondiff":
        residual, scale = ut - .19*uxx - .35*u, ut
    elif system == "nir1_wave":
        residual, scale = utt - .64*uxx, utt
    else:
        residual, scale = ut + .65*ux - .13*uxx, ut
    cropped = np.s_[4:-4, 4:-4]
    assert _error(residual[cropped], scale[cropped]) < .04
