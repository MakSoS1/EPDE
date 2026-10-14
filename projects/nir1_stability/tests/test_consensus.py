"""Truth-free structural-ensemble invariants, including missing/failed seeds."""

import copy

import pytest

from projects.nir1_stability.nir1.consensus import consensus_decision


T = "2.0 * u{power: 1.0} + 0.0 = du/dx0{power: 1.0}"
F = "3.0 * u{power: 2.0} + 0.0 = du/dx0{power: 1.0}"
G = "3.0 * u{power: 3.0} + 0.0 = du/dx0{power: 1.0}"


def _record(seed, equation):
    return {"status": "ok", "seed": seed, "dataset": "ode", "noise": 0.,
            "config": {"nir1_data_seed": 11}, "search_config": {"evolution": {
                "population_size": 16, "training_epochs": 5}},
            "environment": {"epde_commit": "scientific_sha"},
            "front": [[equation]], "objectives": [[0.2, 0.4]],
            "metrics": {"selected_index": 0, "success_selected": False},
            "problem": {"truth": ["spurious value supplied as red herring"]}}


def test_majority_beats_noisy_single_seed_without_oracle():
    data = [_record(0, F), _record(1, T), _record(2, G), _record(3, T), _record(4, F)]
    assert consensus_decision(data)["tie_or_no_repeat"]
    data[-1] = _record(4, G)
    assert consensus_decision(data)["tie_or_no_repeat"]
    data[-1] = _record(4, T)
    outcome = consensus_decision(data)
    assert outcome["mode_used"] and outcome["mode_frequency"] == 3
    assert outcome["chosen_seed"] == 1
    assert outcome["chosen_equations"] == [T]
    # Result must not change if the simulator truth or all offline labels are poisoned.
    poisoned = copy.deepcopy(data)
    for record in poisoned:
        record["problem"]["truth"] = [F]
        record["metrics"]["success_selected"] = not record["metrics"]["success_selected"]
    assert consensus_decision(poisoned) == outcome


def test_no_repeat_and_tie_choose_lowest_optimizer_seed():
    no_repeat = [_record(3, T), _record(0, F), _record(1, G)]
    answer = consensus_decision(no_repeat)
    assert answer["chosen_seed"] == 0 and answer["tie_or_no_repeat"]
    tied = [_record(3, T), _record(0, F), _record(1, T), _record(2, F)]
    answer = consensus_decision(tied)
    assert answer["chosen_seed"] == 0
    assert answer["mode_frequency"] == 2 and not answer["mode_used"]


@pytest.mark.parametrize("change", ["status", "seed", "dataset", "data_seed",
                                    "evolution", "commit", "selected"])
def test_incompatible_or_failed_restarts_rejected(change):
    data = [_record(0, F), _record(1, T), _record(2, T)]
    if change == "status":
        data[0]["status"] = "error"
    elif change == "seed":
        data[1]["seed"] = 0
    elif change == "dataset":
        data[1]["dataset"] = "vdp"
    elif change == "data_seed":
        data[1]["config"]["nir1_data_seed"] = 12
    elif change == "evolution":
        data[1]["search_config"]["evolution"]["population_size"] = 4
    elif change == "commit":
        data[1]["environment"]["epde_commit"] = "other"
    elif change == "selected":
        data[1]["metrics"]["selected_index"] = 4
    with pytest.raises(ValueError):
        consensus_decision(data)


def test_equivalent_equations_ignore_coefficients_and_target_orientation():
    a = _record(2, "1.0 * u{power: 1.0} + 0.0 = du/dx0{power: 1.0}")
    b = _record(1, "4.0 * du/dx0{power: 1.0} + 0.0 = u{power: 1.0}")
    outcome = consensus_decision([a,b])
    assert outcome["mode_used"]
    assert outcome["distinct_structures"] == 1


def test_equal_compute_min_discrepancy_is_truth_free_and_uses_all_runs():
    from projects.nir1_stability.nir1.consensus import min_discrepancy_restart
    records = [_record(0, F), _record(1, T), _record(2, F)]
    records[0]["objectives"] = [[0.9, 0.1]]
    records[1]["objectives"] = [[0.1, 0.8]]
    records[2]["objectives"] = [[0.4, 0.2]]
    pick = min_discrepancy_restart(records)
    assert pick["chosen_seed"] == 1
    assert pick["n_full_searches"] == 3
    assert pick["chosen_equations"] == [T]
    assert pick["truth_used_to_select"] is False
    flipped = copy.deepcopy(records)
    for row in flipped:
        row["problem"] = {"truth": [F]}
        row["metrics"]["success_selected"] = True
    assert min_discrepancy_restart(flipped) == pick


def test_equal_compute_discrepancy_rejects_invalid_objectives():
    from projects.nir1_stability.nir1.consensus import min_discrepancy_restart
    records = [_record(0, F), _record(1, T)]
    records[0]["objectives"] = [[float("nan"), 0.]]
    with pytest.raises(ValueError, match="Nonfinite"):
        min_discrepancy_restart(records)
