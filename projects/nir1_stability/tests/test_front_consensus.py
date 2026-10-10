"""Frozen secondary multi-front selection has no truth leakage."""

import copy
import pytest

from projects.nir1_stability.nir1.front_consensus import pool_consensus_decision

A = "1.0 * u{power: 1.0} = du/dx0{power: 1.0}"
B = "1.0 * u{power: 2.0} = du/dx0{power: 1.0}"
C = "1.0 * u{power: 3.0} = du/dx0{power: 1.0}"
D = "1.0 * u{power: 4.0} = du/dx0{power: 1.0}"


def record(seed, front, selected=0, discrepancies=None):
    if discrepancies is None:
        discrepancies = [i / 10 for i in range(len(front))]
    return {
        "status": "ok", "seed": seed, "dataset": "ode",
        "noise": 0., "config": {"nir1_data_seed": 10},
        "search_config": {"evolution": {"population_size": 16, "training_epochs": 5}},
        "environment": {"epde_commit": "frozen"},
        "front": [[s] for s in front],
        "objectives": [[float(loss), .3] for loss in discrepancies],
        "metrics": {"selected_index": selected, "success_selected": False},
        "problem": {"truth": [A]},
    }


def test_rescue_repeated_front_candidate_never_selected_by_pic():
    rows = [record(0,[B,A],0), record(1,[C,A],0),
            record(2,[D,A],0), record(3,[B],0), record(4,[C],0)]
    result=pool_consensus_decision(rows)
    assert result["chosen_equations"] == [A]
    assert result["chosen_seed"] == 0
    assert result["front_run_frequency"] == 3
    assert result["native_selected_frequency"] == 0
    assert result["allfront_fallback_to_v1"] is False
    assert result["n_full_searches"] == 5
    corrupted=copy.deepcopy(rows)
    for row in corrupted:
        row["problem"]["truth"]=[D]
        row["metrics"]["success_selected"]=True
    assert pool_consensus_decision(corrupted) == result


def test_duplicate_candidate_support_on_one_front_counts_only_one_run():
    rows=[record(0,[A,"20.0 * u{power: 1.0} = du/dx0{power: 1.0}",B]),
          record(1,[C]),record(2,[D])]
    result=pool_consensus_decision(rows)
    assert result["allfront_fallback_to_v1"] is True


def test_frequency_counts_distinct_restarts_not_pareto_rows():
    rows=[record(0,[B,A]),record(1,[C,A]),record(2,[A,D]),
          record(3,[B]),record(4,[C])]
    result=pool_consensus_decision(rows)
    assert result["front_run_frequency"]==3
    assert result["chosen_equations"] == [A]


def test_missing_front_or_failed_search_blocks_whole_group():
    rows=[record(0,[A,B]),record(1,[B,A])]
    bad=copy.deepcopy(rows)
    bad[0]["front"].append([C])
    with pytest.raises(ValueError, match="front"):
        pool_consensus_decision(bad)
    bad=copy.deepcopy(rows)
    bad[1]["status"]="error"
    with pytest.raises(ValueError,match="Failed"):
        pool_consensus_decision(bad)


def test_invalid_objective_or_nonfinite_rejected():
    rows=[record(0,[A,B]),record(1,[B,A])]
    rows[0]["objectives"][0][0] = float("nan")
    with pytest.raises(ValueError,match="Nonfinite"):
        pool_consensus_decision(rows)


def test_front_order_permutation_does_not_change_voted_structure():
    rows=[record(0,[B,A]),record(1,[C,A]),record(2,[D,A])]
    choice=pool_consensus_decision(rows)
    shuffled=copy.deepcopy(rows)
    for row in shuffled:
        row["front"]=row["front"][::-1]
        row["objectives"]=row["objectives"][::-1]
        row["metrics"]["selected_index"]=len(row["front"])-1
    choice2=pool_consensus_decision(shuffled)
    assert choice["chosen_structure_sha256"] == choice2["chosen_structure_sha256"]
    assert choice["chosen_equations"] == choice2["chosen_equations"]


def test_tied_front_frequency_prefers_selected_frequency_not_truth():
    rows=[record(0,[B,A],0),record(1,[C,A],0),
          record(2,[B,C],0),record(3,[D,C],1)]
    result=pool_consensus_decision(rows)
    # C appears in 3 restarts and B in 2, regardless of physical truth.
    assert result["chosen_equations"]==[C]
    assert result["truth_used_to_select"] is False
