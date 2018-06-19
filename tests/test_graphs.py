import pytest
import numpy as np
from causal_learning.dag import DirectedGraph
from causal_learning.utils import create_graph_hyp_space
from causal_learning.graph_teacher import GraphTeacher


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
    t = 0.8  # transmission rate
    b = 0.01  # background rate

    lik = np.array(
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

    graphs = create_graph_hyp_space(t=t, b=b)
    graph_teacher = GraphTeacher(graphs)
    graph_teacher.likelihood()

    assert np.all(np.isclose(graph_teacher.lik, lik))


def test_causal_graph_simulation():
    # matrix of teacher likelihoods for the three causal graphs
    teach_true = np.array([[[0.29915152, 0.14579841, 0.14579841],
                            [0.12994738, 0.04608038, 0.02859807],
                            [0.12994738, 0.02859807, 0.04608038]],

                           [[0.15217036, 0.19220901, 0.07532437],
                            [0.19220901, 0.15217036, 0.07532437],
                            [0.07093121, 0.07093121, 0.01873007]],

                           [[0.1463736, 0.19433433, 0.07421954],
                            [0.24782723, 0.11684906, 0.05856264],
                            [0.05073297, 0.08789775, 0.02320289]]])

    graphs = create_graph_hyp_space(t=0.8, b=0.01)
    graph_teacher = GraphTeacher(graphs)
    graph_teacher.likelihood()
    graph_teacher.update_teacher_posterior(graph_teacher.learner_prior)
    graph_teacher.update_learner_posterior()
    graph_teacher.update_sequential_teacher_posterior()
    teach = graph_teacher.teacher_likelihood(
        graph_teacher.teacher_posterior, graph_teacher.sequential_teacher_posterior)

    assert np.all(np.isclose(teach_true, teach))
