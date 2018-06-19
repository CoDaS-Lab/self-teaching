import pytest
import numpy as np
from causal_learning.dag import DirectedGraph


def test_get_parents():
    common_cause = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]])
    dag_one = DirectedGraph(common_cause)

    assert np.array_equal(dag_one.get_parents(0), np.array([]))
    assert np.array_equal(dag_one.get_parents(1), np.array([0]))
    assert np.array_equal(dag_one.get_parents(2), np.array([0]))

    common_effect = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]])
    dag_two = DirectedGraph(common_effect)

    assert np.array_equal(dag_two.get_parents(0), np.array([]))
    assert np.array_equal(dag_two.get_parents(1), np.array([]))
    assert np.array_equal(dag_two.get_parents(2), np.array([0, 1]))

    causal_chain = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    dag_three = DirectedGraph(causal_chain)

    assert np.array_equal(dag_three.get_parents(0), np.array([]))
    assert np.array_equal(dag_three.get_parents(1), np.array([0]))
    assert np.array_equal(dag_three.get_parents(2), np.array([1]))


def test_get_children():
    common_cause = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]])
    dag_one = DirectedGraph(common_cause)

    assert np.array_equal(dag_one.get_children(0), np.array([1, 2]))
    assert np.array_equal(dag_one.get_children(1), np.array([]))
    assert np.array_equal(dag_one.get_children(2), np.array([]))

    common_effect = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]])
    dag_two = DirectedGraph(common_effect)

    assert np.array_equal(dag_two.get_children(0), np.array([2]))
    assert np.array_equal(dag_two.get_children(1), np.array([2]))
    assert np.array_equal(dag_two.get_children(2), np.array([]))

    causal_chain = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    dag_three = DirectedGraph(causal_chain)

    assert np.array_equal(dag_three.get_children(0), np.array([1]))
    assert np.array_equal(dag_three.get_children(1), np.array([2]))
    assert np.array_equal(dag_three.get_children(2), np.array([]))


def calculate_likelihood(outcomes):
    # use outcomes matrix to determine likelihood
    observations = np.array([[0, 0, 0], [0, 0, 1],
                             [0, 1, 0], [0, 1, 1],
                             [1, 0, 0], [1, 0, 1],
                             [1, 1, 0], [1, 1, 1]])
    n_observations = 8
    n_actions = 3
    lik = np.zeros((n_observations, n_actions))

    for i in range(n_observations):
        for j in range(n_actions):
            observation = observations[i]
            outcome = outcomes[j]
            new_outcome = np.zeros_like(outcome)

            for k, o in enumerate(observation):
                if np.isclose(o, 0):
                    new_outcome[k] = 1 - outcome[k]
                else:
                    new_outcome[k] = outcome[k]

            lik[i, j] = np.prod(new_outcome)

    # check likelihoods for each action sum to 1
    np.all(np.sum(lik, axis=0) == 1)
    return lik


def test_likelihood():
    t = 0.8  # transmission rate
    b = 0.01  # background rate

    pDgHI = np.array(
        [[0.03920400, 0.98010000, 0.98010000, 0.19602000, 0.19602000, 0.98010000,
          0.19602000, 0.19602000, 0.98010000, 0.19602000, 0.19602000, 0.98010000],
         [0.15879600, 0.00990000, 0.00198000, 0.79398000, 0.00039600, 0.00990000,
          0.00198000, 0.15879600, 0.00990000, 0.79398000, 0.00198000, 0.00198000],
         [0.15879600, 0.00198000, 0.00990000, 0.00039600, 0.79398000, 0.00990000,
          0.15879600, 0.00198000, 0.00198000, 0.00198000, 0.79398000, 0.00990000],
         [0.64320400, 0.00802000, 0.00802000, 0.00960400, 0.00960400, 0.00010000,
          0.64320400, 0.64320400, 0.00802000, 0.00802000, 0.00802000, 0.00802000],
         [0.98010000, 0.03920400, 0.98010000, 0.19602000, 0.98010000, 0.19602000,
          0.19602000, 0.98010000, 0.19602000, 0.19602000, 0.98010000, 0.19602000],
         [0.00990000, 0.15879600, 0.00198000, 0.79398000, 0.00990000, 0.00039600,
          0.79398000, 0.00990000, 0.15879600, 0.00198000, 0.00198000, 0.00198000],
         [0.98010000, 0.98010000, 0.03920400, 0.98010000, 0.19602000, 0.19602000,
          0.98010000, 0.19602000, 0.19602000, 0.98010000, 0.19602000, 0.19602000],
         [0.00990000, 0.00198000, 0.15879600, 0.00990000, 0.79398000, 0.00039600,
          0.00990000, 0.79398000, 0.00198000, 0.00198000, 0.00198000, 0.15879600],
         [0.00198000, 0.15879600, 0.00990000, 0.00039600, 0.00990000, 0.79398000,
          0.00198000, 0.00198000, 0.00198000, 0.15879600, 0.00990000, 0.79398000],
         [0.00802000, 0.64320400, 0.00802000, 0.00960400, 0.00010000, 0.00960400,
          0.00802000, 0.00802000, 0.64320400, 0.64320400, 0.00802000, 0.00802000],
         [0.00198000, 0.00990000, 0.15879600, 0.00990000, 0.00039600, 0.79398000,
          0.00198000, 0.00198000, 0.79398000, 0.00990000, 0.15879600, 0.00198000],
         [0.00802000, 0.00802000, 0.64320400, 0.00010000, 0.00960400, 0.00960400,
          0.00802000, 0.00802000, 0.00802000, 0.00802000, 0.64320400, 0.64320400]
         ])

    observations = np.array([[0, 0, 0], [0, 0, 1],
                             [0, 1, 0], [0, 1, 1],
                             [1, 0, 0], [1, 0, 1],
                             [1, 1, 0], [1, 1, 1]])

    common_cause = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]])
    dag_one = DirectedGraph(common_cause, transmission_rate=t,
                            background_rate=b)

    # analytical likelihood
    outcomes_dag_one = np.array([[1.0, t+b, t+b],
                                 [b, 1.0, t*b + b],
                                 [b, t*b + b, 1.0]])

    likelihood_dag_one = np.array([[(0*1.0)*(0*(t+b))*(0*(t+b))],
                                   [(0*b)*(0*1.0)*(0*(t*b + b))],
                                   [(0*b)*(0*(t*b + b)*(0*1.0))]],
                                  [[], [], []],
                                  [[], [], []],
                                  [[], [], []],
                                  [[], [], []],
                                  [[], [], []],
                                  [[], [], []],
                                  [[], [], []])

    likelihood_dag_one = calculate_likelihood(outcomes_dag_one)

    assert np.array_equal(dag_one.likelihood(), likelihood_dag_one)

    common_effect = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]])
    dag_two = DirectedGraph(common_effect, transmission_rate=t,
                            background_rate=b)

    outcomes_dag_two = np.array([[1.0, b, t + t*b + b],
                                 [b, 1.0, t + t*b + b],
                                 [b, b, 1.0]])

    likelihood_dag_two = calculate_likelihood(outcomes_dag_two)

    assert np.array_equal(dag_two.likelihood(), likelihood_dag_two)

    causal_chain = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    dag_three = DirectedGraph(causal_chain, transmission_rate=t,
                              background_rate=b)

    outcomes_dag_three = np.array([[1.0, t+b, t*(t+b) + b],
                                   [b, 1.0, t+b],
                                   [b, t*b + b, 1.0]])

    likelihood_dag_three = calculate_likelihood(outcomes_dag_three)

    assert np.array_equal(dag_three.likelihood(), likelihood_dag_three)
