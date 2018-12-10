import pytest
import numpy as np
from models.utils import create_line_hyp_space
from models.utils import create_boundary_hyp_space
from models.concept_active_learner import ConceptActiveLearner
from models.concept_self_teacher import ConceptSelfTeacher


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
    true_hyp_space_one = np.array([[1, 1], [1, 0], [0, 0]])

    assert np.array_equal(boundary_hyp_space_one, true_hyp_space_one)

    n_features = 4
    boundary_hyp_space_two = create_boundary_hyp_space(n_features)
    true_hyp_space_two = np.array([[1, 1, 1, 1],
                                   [1, 1, 1, 0],
                                   [1, 1, 0, 0],
                                   [1, 0, 0, 0],
                                   [0, 0, 0, 0]])

    assert np.array_equal(boundary_hyp_space_two, true_hyp_space_two)


def test_concept_self_teacher():
    n_features = 3
    hyp_space_type = "boundary"

    self_teacher = ConceptSelfTeacher(n_features, hyp_space_type)
    self_teacher.update_learner_posterior()
    self_teacher.update_self_teaching_posterior()

    first_feature_prob = np.array([50/154, 54/154, 50/154])

    assert np.allclose(first_feature_prob,
                       self_teacher.self_teaching_posterior)


def test_concept_active_learner():
    n_features = 3
    hyp_space_type = "boundary"

    active_learner = ConceptActiveLearner(n_features, hyp_space_type)
    active_learner.update_posterior()
    active_learner_prob = active_learner.expected_information_gain()

    first_feature_prob = np.array([np.log2(4) - 3/4 * np.log2(3),
                                   np.log2(4) - np.log2(2),
                                   np.log2(4) - 3/4 * np.log2(3)])

    assert np.allclose(first_feature_prob,
                       active_learner_prob)
