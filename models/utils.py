import numpy as np
from models import dag


def create_line_hyp_space(n_features):
    """Creates a hypothesis space of concepts defined by 1D lines"""
    hyp_space = []
    for i in range(1, n_features + 1):
        for j in range(n_features - i + 1):
            hyp = [0 for _ in range(n_features)]
            hyp[j:j + i] = [1 for _ in range(i)]
            hyp_space.append(hyp)
    hyp_space = np.array(hyp_space)
    return hyp_space


def create_boundary_hyp_space(n_features):
    """Creates a hypothesis space of concepts defined by a linear boundary"""
    hyp_space = []
    for i in range(n_features + 1):
        hyp = [1 for _ in range(n_features)]
        hyp[n_features-i:n_features] = [0 for _ in range(i)]
        hyp_space.append(hyp)
    hyp_space = np.array(hyp_space)
    return hyp_space


def create_graph_hyp_space(t=0.8, b=0.01):
    """Creates a dict containing all possible common cause, common effect,
    causal chain and single link graphs, along with their likelihoods"""

    # enumerate all graphs with three nodes
    common_cause_1 = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]])
    common_cause_1_cpd = [np.array([1-b, b]),
                          np.array([[1-b, b], [(1-t)*(1-b), t + (1-t)*b]]),
                          np.array([[1-b, b], [(1-t)*(1-b), t + (1-t)*b]])]

    common_cause_2 = np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0]])
    common_cause_2_cpd = [common_cause_1_cpd[i] for i in [1, 0, 2]]

    common_cause_3 = np.array([[0, 0, 0], [0, 0, 0], [1, 1, 0]])
    common_cause_3_cpd = [common_cause_1_cpd[i] for i in [1, 2, 0]]

    common_effect_1 = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]])
    common_effect_1_cpd = [np.array([1-b, b]),
                           np.array([1-b, b]),
                           np.array([[[1-b, b], [(1-t)*(1-b), (t + (1-t)*b)]],
                                     [[(1-t)*(1-b), (t + (1-t)*b)],
                                      [(1-t)*(1-b)*(1-t),
                                       (t)**2 + (t*(1-t))*2 + b*(1-t)**2]]])]

    common_effect_2 = np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]])
    common_effect_2_cpd = [common_effect_1_cpd[i] for i in [0, 2, 1]]

    common_effect_3 = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]])
    common_effect_3_cpd = [common_effect_1_cpd[i] for i in [2, 0, 1]]

    causal_chain_1 = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    causal_chain_1_cpd = [np.array([1-b, b]),
                          np.array([[1-b, b], [(1-t)*(1-b), t + (1-t)*b]]),
                          np.array([[1-b, b], [(1-t)*(1-b), t + (1-t)*b]])]

    causal_chain_2 = np.array([[0, 0, 1], [0, 0, 0], [0, 1, 0]])
    causal_chain_2_cpd = [causal_chain_1_cpd[i] for i in [0, 2, 1]]

    causal_chain_3 = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0]])
    causal_chain_3_cpd = [causal_chain_1_cpd[i] for i in [1, 0, 2]]

    causal_chain_4 = np.array([[0, 0, 1], [1, 0, 0], [0, 0, 0]])
    causal_chain_4_cpd = [causal_chain_1_cpd[i] for i in [2, 0, 1]]

    causal_chain_5 = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0]])
    causal_chain_5_cpd = [causal_chain_1_cpd[i] for i in [1, 2, 0]]

    causal_chain_6 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    causal_chain_6_cpd = [causal_chain_1_cpd[i] for i in [2, 1, 0]]

    single_link_1 = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
    single_link_1_cpd = [np.array([1-b, b]),
                         np.array([[1-b, b], [(1-t)*(1-b), t + (1-t)*b]]),
                         np.array([1-b, b])]

    single_link_2 = np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]])
    single_link_2_cpd = [single_link_1_cpd[i] for i in [0, 2, 1]]

    single_link_3 = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]])
    single_link_3_cpd = [single_link_1_cpd[i] for i in [1, 0, 2]]

    single_link_4 = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
    single_link_4_cpd = [single_link_1_cpd[i] for i in [2, 0, 1]]

    single_link_5 = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]])
    single_link_5_cpd = [single_link_1_cpd[i] for i in [1, 2, 0]]

    single_link_6 = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]])
    single_link_6_cpd = [single_link_1_cpd[i] for i in [2, 1, 0]]

    graph_names = ["common_cause_1", "common_cause_2", "common_cause_3",
                   "common_effect_1", "common_effect_2", "common_effect_3",
                   "causal_chain_1", "causal_chain_2", "causal_chain_3",
                   "causal_chain_4", "causal_chain_5", "causal_chain_6",
                   "single_link_1", "single_link_2", "single_link_3",
                   "single_link_4", "single_link_5", "single_link_6"]

    graphs = [common_cause_1, common_cause_2, common_cause_3,
              common_effect_1, common_effect_2, common_effect_3,
              causal_chain_1, causal_chain_2, causal_chain_3,
              causal_chain_4, causal_chain_5, causal_chain_6,
              single_link_1, single_link_2, single_link_3,
              single_link_4, single_link_5, single_link_6]

    cpds = [common_cause_1_cpd, common_cause_2_cpd, common_cause_3_cpd,
            common_effect_1_cpd, common_effect_2_cpd, common_effect_3_cpd,
            causal_chain_1_cpd, causal_chain_2_cpd, causal_chain_3_cpd,
            causal_chain_4_cpd, causal_chain_5_cpd, causal_chain_6_cpd,
            single_link_1_cpd, single_link_2_cpd, single_link_3_cpd,
            single_link_4_cpd, single_link_5_cpd, single_link_6_cpd]

    hyp_space = {graph_names: dag.DirectedGraph(graph, cpd, t, b)
                 for (graph_names, graph, cpd) in
                 zip(graph_names, graphs, cpds)}

    return hyp_space


def create_teaching_hyp_space(t=0.8, b=0.1):
    hyp_space = create_graph_hyp_space(t=t, b=b)

    teaching_hyps = ['common_cause_1', 'common_cause_2', 'common_cause_3',
                     'common_effect_1', 'common_effect_2', 'common_effect_3',
                     'causal_chain_1', 'causal_chain_2', 'causal_chain_3',
                     'causal_chain_4', 'causal_chain_5', 'causal_chain_6']

    teaching_hyp_space = [hyp_space[hyps] for hyps in teaching_hyps]

    return teaching_hyp_space


def create_active_learning_hyp_space(t=0.8, b=0.0):
    hyp_space = create_graph_hyp_space(t=t, b=b)

    problem_1_graphs = [hyp_space[hyps]
                        for hyps in ['common_cause_2', 'common_cause_1']]
    problem_2_graphs = [hyp_space[hyps]
                        for hyps in ['common_cause_2', 'common_effect_2']]
    problem_3_graphs = [hyp_space[hyps]
                        for hyps in ['common_cause_2', 'causal_chain_1']]
    problem_4_graphs = [hyp_space[hyps]
                        for hyps in ['common_cause_2', 'causal_chain_4']]
    problem_5_graphs = [hyp_space[hyps]
                        for hyps in ['common_cause_2', 'causal_chain_2']]
    problem_6_graphs = [hyp_space[hyps]
                        for hyps in ['common_cause_2', 'single_link_1']]
    problem_7_graphs = [hyp_space[hyps]
                        for hyps in ['common_cause_2', 'single_link_2']]
    problem_8_graphs = [hyp_space[hyps]
                        for hyps in ['common_cause_2', 'single_link_3']]
    problem_9_graphs = [hyp_space[hyps]
                        for hyps in ['common_cause_1', 'common_effect_2']]
    problem_10_graphs = [hyp_space[hyps]
                         for hyps in ['common_effect_2', 'common_effect_3']]
    problem_11_graphs = [hyp_space[hyps]
                         for hyps in ['common_effect_2', 'causal_chain_1']]
    problem_12_graphs = [hyp_space[hyps]
                         for hyps in ['common_effect_2', 'causal_chain_4']]
    problem_13_graphs = [hyp_space[hyps]
                         for hyps in ['common_effect_2', 'causal_chain_2']]
    problem_14_graphs = [hyp_space[hyps]
                         for hyps in ['common_effect_2', 'single_link_1']]
    problem_15_graphs = [hyp_space[hyps]
                         for hyps in ['common_effect_2', 'single_link_2']]
    problem_16_graphs = [hyp_space[hyps]
                         for hyps in ['common_effect_2', 'single_link_3']]
    problem_17_graphs = [hyp_space[hyps]
                         for hyps in ['causal_chain_1', 'causal_chain_3']]
    problem_18_graphs = [hyp_space[hyps]
                         for hyps in ['causal_chain_1', 'causal_chain_4']]
    problem_19_graphs = [hyp_space[hyps]
                         for hyps in ['causal_chain_1', 'causal_chain_6']]
    problem_20_graphs = [hyp_space[hyps]
                         for hyps in ['causal_chain_1', 'causal_chain_5']]
    problem_21_graphs = [hyp_space[hyps]
                         for hyps in ['causal_chain_1', 'causal_chain_2']]
    problem_22_graphs = [hyp_space[hyps]
                         for hyps in ['causal_chain_1', 'single_link_1']]
    problem_23_graphs = [hyp_space[hyps]
                         for hyps in ['causal_chain_1', 'single_link_2']]
    problem_24_graphs = [hyp_space[hyps]
                         for hyps in ['causal_chain_1', 'single_link_3']]
    problem_25_graphs = [hyp_space[hyps]
                         for hyps in ['causal_chain_1', 'single_link_4']]
    problem_26_graphs = [hyp_space[hyps]
                         for hyps in ['causal_chain_1', 'single_link_5']]
    problem_27_graphs = [hyp_space[hyps]
                         for hyps in ['causal_chain_1', 'single_link_6']]

    active_learning_problems = [problem_1_graphs, problem_2_graphs,
                                problem_3_graphs, problem_4_graphs,
                                problem_5_graphs, problem_6_graphs,
                                problem_7_graphs, problem_8_graphs,
                                problem_9_graphs, problem_10_graphs,
                                problem_11_graphs, problem_12_graphs,
                                problem_13_graphs, problem_14_graphs,
                                problem_15_graphs, problem_16_graphs,
                                problem_17_graphs, problem_18_graphs,
                                problem_19_graphs, problem_20_graphs,
                                problem_21_graphs, problem_22_graphs,
                                problem_23_graphs, problem_24_graphs,
                                problem_25_graphs, problem_26_graphs,
                                problem_27_graphs]

    return active_learning_problems
