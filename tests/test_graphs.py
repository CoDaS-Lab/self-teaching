import pytest
import numpy as np
from causal_learning import utils
from causal_learning.dag import DirectedGraph
from causal_learning.graph_teacher import GraphTeacher
from causal_learning.graph_active_learner import GraphActiveLearner


def test_get_parents():
    hyp_space = utils.create_graph_hyp_space()

    dag_one = hyp_space["common_cause_1"]

    assert np.array_equal(dag_one.get_parents(0, dag_one.graph), np.array([]))
    assert np.array_equal(dag_one.get_parents(1, dag_one.graph), np.array([0]))
    assert np.array_equal(dag_one.get_parents(2, dag_one.graph), np.array([0]))

    dag_two = hyp_space["common_effect_1"]

    assert np.array_equal(dag_two.get_parents(0, dag_two.graph), np.array([]))
    assert np.array_equal(dag_two.get_parents(1, dag_two.graph), np.array([]))
    assert np.array_equal(dag_two.get_parents(
        2, dag_two.graph), np.array([0, 1]))

    dag_three = hyp_space["causal_chain_1"]

    assert np.array_equal(dag_three.get_parents(
        0, dag_three.graph), np.array([]))
    assert np.array_equal(dag_three.get_parents(
        1, dag_three.graph), np.array([0]))
    assert np.array_equal(dag_three.get_parents(
        2, dag_three.graph), np.array([1]))


def test_get_children():
    hyp_space = utils.create_graph_hyp_space()

    dag_one = hyp_space["common_cause_1"]

    assert np.array_equal(dag_one.get_children(
        0, dag_one.graph), np.array([1, 2]))
    assert np.array_equal(dag_one.get_children(1, dag_one.graph), np.array([]))
    assert np.array_equal(dag_one.get_children(2, dag_one.graph), np.array([]))

    dag_two = hyp_space["common_effect_1"]

    assert np.array_equal(dag_two.get_children(
        0, dag_two.graph), np.array([2]))
    assert np.array_equal(dag_two.get_children(
        1, dag_two.graph), np.array([2]))
    assert np.array_equal(dag_two.get_children(2, dag_two.graph), np.array([]))

    dag_three = hyp_space["causal_chain_1"]

    assert np.array_equal(dag_three.get_children(
        0, dag_three.graph), np.array([1]))
    assert np.array_equal(dag_three.get_children(
        1, dag_three.graph), np.array([2]))
    assert np.array_equal(dag_three.get_children(
        2, dag_three.graph), np.array([]))


def test_intervene():
    t = 0.8
    b = 0.01
    hyp_space = utils.create_graph_hyp_space(t, b)

    common_effect_1 = hyp_space["common_effect_1"]

    common_effect_intervene_0 = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]])
    common_effect_intervene_1 = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]])
    common_effect_intervene_2 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

    common_effect_1.intervene(0)
    assert np.array_equal(
        common_effect_1.intervened_graph, common_effect_intervene_0)

    common_effect_1.intervene(1)
    assert np.array_equal(
        common_effect_1.intervened_graph, common_effect_intervene_1)

    common_effect_1.intervene(2)
    assert np.array_equal(
        common_effect_1.intervened_graph, common_effect_intervene_2)

    causal_chain_1 = hyp_space["causal_chain_1"]

    causal_chain_intervene_0 = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    causal_chain_intervene_1 = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
    causal_chain_intervene_2 = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])

    causal_chain_1.intervene(0)
    assert np.array_equal(
        causal_chain_1.intervened_graph, causal_chain_intervene_0)

    causal_chain_1.intervene(1)
    assert np.array_equal(
        causal_chain_1.intervened_graph, causal_chain_intervene_1)

    causal_chain_1.intervene(2)
    assert np.array_equal(
        causal_chain_1.intervened_graph, causal_chain_intervene_2)


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

    graphs = utils.create_teaching_hyp_space(t=t, b=b)
    graph_teacher = GraphTeacher(graphs)
    print(graph_teacher.likelihood())

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

    graphs = utils.create_teaching_hyp_space(t=0.8, b=0.01)
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
    hyp_space = utils.create_graph_hyp_space(t=t, b=b)
    example_one_graph_names = ['common_cause_1', 'common_cause_2']
    example_one_graphs = [hyp_space[graph_name]
                          for graph_name in example_one_graph_names]

    gal_one = GraphActiveLearner(example_one_graphs)
    gal_one.update_posterior()

    true_eig = np.array([0.5, 0.5, 0])

    assert np.array_equal(true_eig, gal_one.expected_information_gain())


def test_graph_active_learner_two():
    t = 0.8  # transmission rate
    b = 0.0  # background rate

    hyp_space = utils.create_graph_hyp_space(t=t, b=b)
    example_two_graph_names = ['causal_chain_1', 'causal_chain_2']
    example_two_graphs = [hyp_space[graph_name]
                          for graph_name in example_two_graph_names]

    gal_two = GraphActiveLearner(example_two_graphs)
    gal_two.update_posterior()

    true_eig = np.array([0.11594429, 0.44202786, 0.44202786])

    assert np.allclose(true_eig, gal_two.expected_information_gain())


def test_graph_active_learner_three():
    t = 0.8  # transmission rate
    b = 0.0  # background rate

    hyp_space = utils.create_graph_hyp_space(t=t, b=b)
    example_three_graph_names = ['common_effect_2', 'single_link_1']
    example_three_graphs = [hyp_space[graph_name]
                            for graph_name in example_three_graph_names]

    gal_three = GraphActiveLearner(example_three_graphs)
    gal_three.update_posterior()

    true_eig = np.array([0.0, 0.0, 1.0])

    assert np.allclose(true_eig, gal_three.expected_information_gain())
