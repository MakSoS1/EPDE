"""Regress the two genuine cached tensor representations of one domain."""

import numpy as np

from epde.structure.domain import Trajectory


class DummyCache:
    def __init__(self, array):
        self.array=array
    def get(self, *, subcache_ID):
        return self.array


class DummyDomain:
    inner_shape=np.array([65,38])


def _trial(array):
    trajectory=Trajectory.__new__(Trajectory)
    trajectory._cache=DummyCache(array)
    trajectory._cache_id=101
    return trajectory.checkShapes(DummyDomain())


def test_same_inner_grid_shape_accepted():
    assert _trial(np.zeros((65,38)))


def test_identical_point_count_stored_as_flattened_vector_accepted():
    assert _trial(np.zeros(65*38))


def test_incompatible_inner_grid_length_rejected():
    assert not _trial(np.zeros(65*38+1))


def test_incompatible_geometry_even_if_total_count_same_rejected():
    # The flat representation is unavoidable, but 2D representation must
    # preserve its axis geometry rather than only checking product.
    assert not _trial(np.zeros((38,65)))
