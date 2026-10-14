"""Paired front replay must not rerun evolution or use truth for selection."""

from copy import deepcopy
import pytest

from projects.nir1_stability.nir1.paired_replay import paired_selector_replay


def _native():
    a = ("0.001 * d^4u/dx1^4{power: 1.0} + "
         "0.1 * d^2u/dx1^2{power: 1.0} = du/dx0{power: 1.0}")
    b = "0.1 * d^2u/dx1^2{power: 1.0} = du/dx0{power: 1.0}"
    return {"status": "ok", "dataset": "test", "seed": 3,
            "front": [[a], [b]], "objectives": [[0.001, 0.02], [0.002, 0.04]],
            "metrics": {"selected_index": 0, "success_selected": False,
                        "hamming_selected": 1},
            "environment": {"epde_commit": "abc123"}}, [b]


def test_replay_pairs_one_immutable_front_with_truth_after_choice():
    source, truth = _native()
    before = deepcopy(source)
    report = paired_selector_replay(source, truth_systems=[truth])
    assert source == before
    assert report["same_front_for_both"] is True
    assert report["evolutions_per_pair"] == 1
    assert report["native_pic"]["selected_index"] == 0
    assert report["sparsefront_v2"]["selected_index"] == 1
    assert report["sparsefront_v2"]["success_selected"] is True
    assert report["original_epde_code_sha"] == "abc123"


def test_replay_cannot_read_truth_during_candidate_selection():
    source, truth = _native()
    no_truth = paired_selector_replay(source, truth_systems=None)
    with_truth = paired_selector_replay(source, truth_systems=[truth])
    with_false_truth = paired_selector_replay(source, truth_systems=[source["front"][0]])
    assert no_truth["sparsefront_v2"]["selected_index"] == with_truth["sparsefront_v2"]["selected_index"]
    assert with_truth["sparsefront_v2"]["selected_index"] == with_false_truth["sparsefront_v2"]["selected_index"]
    assert no_truth["source_front_sha256"] == with_truth["source_front_sha256"]
    assert no_truth["sparsefront_v2"]["success_selected"] is None


def test_replay_rejects_failed_and_unpaired_inputs():
    source, _ = _native()
    source["status"] = "error"
    with pytest.raises(ValueError, match="completed"):
        paired_selector_replay(source, truth_systems=None)
    source["status"] = "ok"
    source["metrics"].pop("selected_index")
    with pytest.raises(ValueError, match="Original PIC"):
        paired_selector_replay(source, truth_systems=None)
