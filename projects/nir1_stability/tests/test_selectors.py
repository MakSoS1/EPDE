import numpy as np

from projects.nir1_stability.nir1.selectors import select_normalized_utopia


def test_empty_invalid_or_degenerate_front_is_handled_explicitly():
    assert select_normalized_utopia([]) is None
    assert select_normalized_utopia([[2.0, 2.0], [2.0, 2.0]]) == 0
    assert select_normalized_utopia([[float('nan'), 2], [3, 4]]) == 1
    assert select_normalized_utopia([[float('nan'), 2]]) is None


def test_scale_normalization_not_truth_metadata_or_array_order():
    front = [[0.0, 9.0], [4.0, 4.0], [9.0, 0.0]]
    assert select_normalized_utopia(front) == 1
    assert select_normalized_utopia(np.array(front) * [1000, 0.0001]) == 1
    assert select_normalized_utopia(front, truth_labels=[False, True, False]) == 1
    assert select_normalized_utopia(front, truth_labels=[True, False, True]) == 1


def test_exact_ties_lexicographically_first_and_zero_range_axes_ignored():
    assert select_normalized_utopia([[1, 0], [1, 5], [1, 10]]) == 0
    assert select_normalized_utopia([[1, 2], [1, 2], [2, 3]]) == 0
