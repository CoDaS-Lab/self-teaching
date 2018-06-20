import pytest
import numpy as np
from causal_learning import utils
from causal_learning.dag import DirectedGraph
from causal_learning.graph_teacher import GraphTeacher
from causal_learning.graph_active_learner import GraphActiveLearner


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

    graphs = utils.create_graph_hyp_space(t=t, b=b)
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

    graphs = utils.create_graph_hyp_space(t=0.8, b=0.01)
    graph_teacher = GraphTeacher(graphs)
    graph_teacher.likelihood()
    graph_teacher.update_teacher_posterior(graph_teacher.learner_prior)
    graph_teacher.update_learner_posterior()
    graph_teacher.update_sequential_teacher_posterior()
    teach = graph_teacher.teacher_likelihood(
        graph_teacher.teacher_posterior, graph_teacher.sequential_teacher_posterior)

    assert np.all(np.isclose(teach_true, teach))


def test_graph_active_learner_one():
    t = 0.8  # transmission rate
    b = 0.0  # background rate

    # example one
    common_cause_1 = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]])
    common_cause_1_lik = np.array(
        [((1-t)*(1-b))**2, (1-t)*(1-b)*(t + (1-t)*b), (t + (1-t)*b)*(1-t)*(1-b), (t + (1-t)*b)**2,
         (1-b)**2, (1-b)*b, (1-b)**2, (1-b)*b,
         b*(1-t)*(1-b), b*(t + (1-t)*b), b*(1-t)*(1-b), b*(t + (1-t)*b)])

    common_cause_2 = np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0]])
    common_cause_2_lik = utils.permute_likelihood(
        common_cause_1_lik, (2, 1, 3))
    common_cause_2_lik[7], common_cause_2_lik[10] = \
        common_cause_2_lik[10], common_cause_2_lik[7]

    graphs_one = [common_cause_2, common_cause_1]
    likelihoods_one = [common_cause_2_lik, common_cause_1_lik]
    common_cause_graphs = [DirectedGraph(graph, likelihood, t, b)
                           for (graph, likelihood) in zip(graphs_one, likelihoods_one)]

    gal_one = GraphActiveLearner(common_cause_graphs)
    gal_one.update_posterior()

    true_eig = np.array([0.5, 0.5, 0])

    assert np.array_equal(true_eig, gal_one.expected_information_gain())


def test_graph_active_learner_two():
    t = 0.8  # transmission rate
    b = 0.0  # background rate

    causal_chain_1 = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    causal_chain_1_lik = np.array(
        [(1-t)*(1-b)*(1-b), (1-t)*(1-b)*b, (t + (1-t)*b)*(1-t)*(1-b), (t + (1-t)*b)**2,
         (1-b)*(1-t)*(1-b), (1-b)*(t + (1-t)*b), (1-b)**2, (1-b)*b,
            b*(1 - t)*(1-b), b*(t + (1-t)*b), b*(1-t)*(1-b), b*(t + (1-t)*b)]
    )

    causal_chain_2 = np.array([[0, 0, 1], [0, 0, 0], [0, 1, 0]])
    causal_chain_2_lik = utils.permute_likelihood(
        causal_chain_1_lik, (1, 3, 2))
    causal_chain_2_lik[1], causal_chain_2_lik[2] = \
        causal_chain_2_lik[2], causal_chain_2_lik[1]

    graphs_two = [causal_chain_2, causal_chain_1]
    likelihoods_two = [causal_chain_2_lik, causal_chain_1_lik]
    causal_chain_graphs = [DirectedGraph(graph, likelihood, t, b)
                           for (graph, likelihood) in zip(graphs_two, likelihoods_two)]

    gal_two = GraphActiveLearner(causal_chain_graphs)
    gal_two.update_posterior()

    true_eig = np.array([0.11594429, 0.44202786, 0.44202786])

    assert np.allclose(true_eig, gal_two.expected_information_gain())
