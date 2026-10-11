"""Two-mode counterexample to claims that H8 universally improves EPDE."""
import pytest

from projects.nir1_stability.nir1.h8_two_mode_audit import (
    PDE_CASES, experiment, one_case, spatial_identity_test,
    two_mode_trajectories,
)


@pytest.mark.parametrize("case", PDE_CASES)
def test_two_mode_still_obeys_exact_physical_law_and_is_identifiable(case):
    row=one_case(case,0)
    assert row["single"]["exact"], "A mixed IC should excite enough features"
    assert row["multi"]["exact"]
    assert abs(row["single"]["training_rows"] -
               row["multi"]["training_rows"])<=3
    assert row["single"]["heldout_relative_mse"]>=0
    assert row["multi"]["heldout_relative_mse"]>=0


def test_spurious_identity_has_zero_error_per_mode_but_fails_mix():
    result=spatial_identity_test()
    assert result["one_mode"]["normalized_rms"] < 1e-12
    assert result["mixed_modes"]["normalized_rms"] > .2


def test_report_discloses_repeated_systems_and_scope():
    result=experiment(range(2))
    assert result["n_physical_families"]==3
    assert result["n_seed_repeats"]==6
    assert "NOT_NEW_CONFIRMATION" in result["status"]
    assert result["all_true_single"]==6
    assert result["all_true_multi"]==6
