"""Strict full EPDE equality must not hide differing candidate multiplicities."""

from copy import deepcopy

import pytest
from projects.nir1_stability.nir1.compare_full_e1 import compare


def _record():
    a = "1.0 * u{power: 1.0} = du/dx0{power: 1.0}"
    b = "2.0 * u{power: 1.0} = du/dx0{power: 1.0}"
    c = "1.0 * du/dx1{power: 1.0} = du/dx0{power: 1.0}"
    return {
        "status": "ok", "dataset": "ode", "seed": 0,
        "config": {"dataset": "ode", "method": "epde"},
        "search_config": {"evolution": {"population_size": 16, "training_epochs": 5}},
        "research_variant": "nir1_protected_regulator",
        "front": [[a], [b], [c]],
        "objectives": [[.1, 2.], [.2, 1.], [.3, .5]],
        "selected": [a],
        "metrics": {"success_selected": False, "success_front": True,
                    "hamming_selected": 1, "hamming_min": 0},
        "total_seconds": 120.,
    }


def test_full_e1_parity_preserves_duplicate_canonical_structure():
    reference = _record()
    cached = deepcopy(reference)
    cached["total_seconds"] = 80.
    result = compare(cached, reference)
    assert result["samples"] == 3
    assert result["same_candidate_support"] is True
    assert result["speed_ratio_one_pair_not_stable_benchmark"] == pytest.approx(1.5)
    cached["objectives"][1][0] += .1
    with pytest.raises(AssertionError, match="Objective changed"):
        compare(cached, reference)


def test_full_e1_parity_rejects_selected_structure_and_budget_drift():
    reference = _record()
    cached = deepcopy(reference)
    cached["selected"] = reference["front"][-1]
    with pytest.raises(AssertionError, match="selection differs"):
        compare(cached, reference)
    cached = deepcopy(reference)
    cached["search_config"]["evolution"]["training_epochs"] = 1
    with pytest.raises(AssertionError, match="configuration"):
        compare(cached, reference)
