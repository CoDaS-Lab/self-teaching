import pytest
import numpy as np
from concept_learning.utils import create_line_hyp_space
from concept_learning.utils import create_boundary_hyp_space


def test_create_line_hyp_space():
    n_features = 2
    line_hyp_space_one = create_line_hyp_space(n_features)
    true_hyp_space_one = np.array([[1, 0], [0, 1], [1, 1]])

    assert np.array_equal(line_hyp_space_one, true_hyp_space_one)

    n_features = 4
    line_hyp_space_two = create_line_hyp_space(n_features)
    true_hyp_space_two = np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1],
                                   [1, 1, 0, 0],
                                   [0, 1, 1, 0],
                                   [0, 0, 1, 1],
                                   [1, 1, 1, 0],
                                   [0, 1, 1, 1],
                                   [1, 1, 1, 1]])

    assert np.array_equal(line_hyp_space_two, true_hyp_space_two)


def test_create_boundary_hyp_space():
    n_features = 2
    boundary_hyp_space_one = create_boundary_hyp_space(n_features)
    true_hyp_space_one = np.array([[1, 1], [0, 1], [0, 0]])

    assert np.array_equal(boundary_hyp_space_one, true_hyp_space_one)

    n_features = 4
    boundary_hyp_space_two = create_boundary_hyp_space(n_features)
    true_hyp_space_two = np.array([[1, 1, 1, 1],
                                   [0, 1, 1, 1],
                                   [0, 0, 1, 1],
                                   [0, 0, 0, 1],
                                   [0, 0, 0, 0]])

    assert np.array_equal(boundary_hyp_space_two, true_hyp_space_two)


# TODO: add test for likelihood

