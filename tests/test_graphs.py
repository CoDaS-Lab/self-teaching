import pytest
import numpy as np
from models.dag import DirectedGraph

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

def test_likelihood():
    t = 0.9  # transmission rate
    b = 0.05  # background rate

    common_cause = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]])
    dag_one = DirectedGraph(common_cause, transmission_rate=t,
                            background_rate=b)

    # analytical likelihood
    likelihood_dag_one = np.array([[1.0, t+b, t+b],
                             [b, 1.0, t*b + b],
                             [b, t*b + b, 1.0]])
    assert np.array_equal(dag_one.likelihood(), likelihood_dag_one)

    common_effect = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]])
    dag_two = DirectedGraph(common_effect, transmission_rate=t,
                            background_rate=b)

    likelihood_dag_two = np.array([[1.0, b, t + t*b + b],
                                   [b, 1.0, t + t*b + b],
                                   [b, b, 1.0]])

    assert np.array_equal(dag_two.likelihood(), likelihood_dag_two)

    causal_chain = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    dag_three = DirectedGraph(causal_chain, transmission_rate=t,
                            background_rate=b)

    likelihood_dag_three = np.array([[1.0, t+b, t*(t+b) + b],
                                     [b, 1.0, t+b],
                                     [b, t*b + b, 1.0]])

    assert np.array_equal(dag_three.likelihood(), likelihood_dag_three)
    
