import hashlib

import numpy as np
import pytest

from projects.nir1_stability.nir1.datasets import (
    check_representability, load_registered_split, pic_dataset_names,
)
from projects.nir1_stability.nir1.fixtures import make_fixture


@pytest.mark.parametrize("case", [f"T{i}" for i in range(1, 9)])
def test_all_cases_are_reproducible_and_have_truth(case):
    left, truth = make_fixture(case, seed=33)
    right, truth_again = make_fixture(case, seed=33)
    assert hashlib.sha256(left.X.tobytes() + left.y.tobytes()).digest() == hashlib.sha256(right.X.tobytes() + right.y.tobytes()).digest()
    assert truth == truth_again
    assert left.X.shape[0] >= 100
    assert left.X.shape[1] == len(truth["beta"])
    assert truth["support"] == [i for i, x in enumerate(truth["beta"]) if abs(x) > 0]


def test_correlation_strata_predeclared():
    corr = []
    for rho in (0.0, 0.5, 0.9, 0.99, 0.999):
        design, truth = make_fixture("T2", 1, correlation=rho)
        corr.append(truth["correlation"])
        assert abs(np.corrcoef(design.X[:, :2].T)[0, 1] - rho) < 0.15
    assert corr == [0., .5, .9, .99, .999]


def test_weak_term_and_scale_stress():
    design, truth = make_fixture("T3")
    assert 1e-4 in np.abs(truth["beta"])
    assert np.ptp(np.linalg.norm(design.X, axis=0)) > 100


def test_variable_class_flag_and_duplicate_rank():
    design, truth = make_fixture("T6")
    assert truth["model_class"] == "variable"
    assert "coefficient_by_environment" in truth
    design, truth = make_fixture("T8")
    assert np.linalg.matrix_rank(design.X) < design.X.shape[1]


def test_noise_in_design_separated_from_response_noise():
    _, truth = make_fixture("T7")
    assert truth["noise_X"] > 0 and truth["noise_y"] > 0


def test_pic_registry_and_disjoint_system_splits():
    names = set(pic_dataset_names())
    assert {"ode", "ac", "burgers", "wave", "ks", "ns"}.issubset(names)
    split = "projects/nir1_stability/configs/splits.yaml"
    dev, val, held = (set(load_registered_split(split, part)) for part in ("development", "validation", "heldout"))
    assert dev and val and held
    assert not (dev & val or dev & held or val & held)
    assert (dev | val | held).issubset(names)


def test_unknown_dataset_is_not_claimed_supported():
    result = check_representability("not_a_dataset", ["u"])
    assert result["status"] == "unsupported"


def test_unavailable_truth_never_claimed_representable():
    result = check_representability("robot_arm", ["u"])
    assert result["status"] != "supported"
