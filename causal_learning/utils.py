import numpy as np
from causal_learning.dag import DirectedGraph


def permute_likelihood(lik, new_lik_order):
    """
    Calculates the likelihood for permuted causal graphs
    Assumes the original order is (1, 2, 3)
    """

    # get likelihood indices for each node
    node_one_idx = [0, 1, 2, 3]
    node_two_idx = [4, 5, 8, 9]
    node_three_idx = [6, 7, 10, 11]
    new_lik = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for i, j in enumerate(new_lik_order):
        original_node_idx = []
        new_node_idx = []

        # figure out indices to use for original and new nodes
        if (i+1) == 1:
            original_node_idx = node_one_idx
        elif (i+1) == 2:
            original_node_idx = node_two_idx
        elif (i+1) == 3:
            original_node_idx = node_three_idx

        if j == 1:
            new_node_idx = node_one_idx
        elif j == 2:
            new_node_idx = node_two_idx
        elif j == 3:
            new_node_idx = node_three_idx

        # use indices to map likelihoods to new causal graph
        for k in range(len(new_node_idx)):
            new_lik[new_node_idx[k]] = lik[original_node_idx[k]]

    return np.array(new_lik)


def create_graph_hyp_space(t=0.8, b=0.01):
    """Creates a dict containing all possible common cause, common effect, causal chain 
    and single link graphs, along with their likelihoods"""

    # enumerate all graphs with three nodes
    common_cause_1 = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]])
    common_cause_1_lik = np.array(
        [((1-t)*(1-b))**2, (1-t)*(1-b)*(t + (1-t)*b), (t + (1-t)*b)*(1-t)*(1-b), (t + (1-t)*b)**2,
         (1-b)**2, (1-b)*b, (1-b)**2, (1-b)*b,
         b*(1-t)*(1-b), b*(t + (1-t)*b), b*(1-t)*(1-b), b*(t + (1-t)*b)])

    common_cause_2 = np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0]])
    common_cause_2_lik = permute_likelihood(common_cause_1_lik, (2, 1, 3))
    common_cause_2_lik[7], common_cause_2_lik[10] = \
        common_cause_2_lik[10], common_cause_2_lik[7]

    common_cause_3 = np.array([[0, 0, 0], [0, 0, 0], [1, 1, 0]])
    common_cause_3_lik = permute_likelihood(common_cause_1_lik, (3, 1, 2))
    common_cause_3_lik[1], common_cause_3_lik[2] = \
        common_cause_3_lik[2], common_cause_3_lik[1]
    common_cause_3_lik[5], common_cause_3_lik[8] = \
        common_cause_3_lik[8], common_cause_3_lik[5]

    common_effect_1 = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]])
    common_effect_1_lik = np.array(
        [(1-b)*(1-t)*(1-b), (1-b)*(t + (1-t)*b), b*(1-t)*(1-t)*(1-b),
         b*(t*(1-t) + (1-t)*t + t*t + b*((1-t)**2)),
         (1-b)*(1-t)*(1-b), (1-b)*(t + (1-t)*b), (1-b)*(1-b), (1-b)*b,
         b*(1-t)*(1-t)*(1-b), b*(t*(1-t) + (1-t)*t + t*t + b*((1-t)**2)), (1-b)*b, b*b])

    common_effect_2 = np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]])
    common_effect_2_lik = permute_likelihood(common_effect_1_lik, (3, 1, 2))
    common_effect_2_lik[1], common_effect_2_lik[2] = \
        common_effect_2_lik[2], common_effect_2_lik[1]

    common_effect_3 = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]])
    common_effect_3_lik = permute_likelihood(common_effect_1_lik, (2, 3, 1))
    common_effect_3_lik[5], common_effect_3_lik[8] = \
        common_effect_3_lik[8], common_effect_3_lik[5]
    common_effect_3_lik[7], common_effect_3_lik[10] = \
        common_effect_3_lik[10], common_effect_3_lik[7]

    causal_chain_1 = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    causal_chain_1_lik = np.array(
        [(1-t)*(1-b)*(1-b), (1-t)*(1-b)*b, (t + (1-t)*b)*(1-t)*(1-b), (t + (1-t)*b)**2,
         (1-b)*(1-t)*(1-b), (1-b)*(t + (1-t)*b), (1-b)**2, (1-b)*b,
         b*(1 - t)*(1-b), b*(t + (1-t)*b), b*(1-t)*(1-b), b*(t + (1-t)*b)]
    )

    causal_chain_2 = np.array([[0, 0, 1], [0, 0, 0], [0, 1, 0]])
    causal_chain_2_lik = permute_likelihood(causal_chain_1_lik, (1, 3, 2))
    causal_chain_2_lik[1], causal_chain_2_lik[2] = \
        causal_chain_2_lik[2], causal_chain_2_lik[1]

    causal_chain_3 = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0]])
    causal_chain_3_lik = permute_likelihood(causal_chain_1_lik, (2, 3, 1))
    causal_chain_3_lik[5], causal_chain_3_lik[8] = \
        causal_chain_3_lik[8], causal_chain_3_lik[5]
    causal_chain_3_lik[7], causal_chain_3_lik[10] = \
        causal_chain_3_lik[10], causal_chain_3_lik[7]

    causal_chain_4 = np.array([[0, 0, 1], [1, 0, 0], [1, 0, 0]])
    causal_chain_4_lik = permute_likelihood(causal_chain_1_lik, (2, 1, 3))
    causal_chain_4_lik[7], causal_chain_4_lik[10] = \
        causal_chain_4_lik[10], causal_chain_4_lik[7]

    causal_chain_5 = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0]])
    causal_chain_5_lik = permute_likelihood(causal_chain_1_lik, (3, 1, 2))
    causal_chain_5_lik[1], causal_chain_5_lik[2] = \
        causal_chain_5_lik[2], causal_chain_5_lik[1]
    causal_chain_5_lik[5], causal_chain_5_lik[8] = \
        causal_chain_5_lik[8], causal_chain_5_lik[5]

    causal_chain_6 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    causal_chain_6_lik = permute_likelihood(causal_chain_1_lik, (3, 2, 1))
    causal_chain_6_lik[1], causal_chain_6_lik[2] = \
        causal_chain_6_lik[2], causal_chain_6_lik[1]
    causal_chain_6_lik[5], causal_chain_6_lik[8] = \
        causal_chain_6_lik[8], causal_chain_6_lik[5]
    causal_chain_6_lik[7], causal_chain_6_lik[10] = \
        causal_chain_6_lik[10], causal_chain_6_lik[7]

    single_link_1 = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
    single_link_lik_1 = np.array([(1-t)*(1-b)*(1-b), (1-t)*(1-b)*b,
                                  (t + (1-t)*b)*(1-b), (t + (1-t)*b)*b,
                                  (1-b)*(1-b), (1-b)*b,
                                  (1-b)*(1-b), (1-b)*b,
                                  b*(1-b), b*b,
                                  b*(1-t)*(1-b), b*(t + (1-t)*b)])

    single_link_2 = np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]])
    single_link_lik_2 = np.array([(1-t)*(1-b)*(1-b), (t + (1-t)*b)*(1-b), b*(1-t)*(1-b),
                                  b*(t + (1-t)*b), (1-b)*(1-b), (1-b)*b,
                                  (1-b)*(1-b), (1-b)*b, b*(1-t)*(1-b),
                                  b*(t + (1-t)*b), b*(1-b), b*b])

    single_link_3 = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]])
    single_link_lik_3 = np.array([(1-b)*(1-b), (1-b)*b, b*(1-b),
                                  b*b, (1-t)*(1-b)*(1-b), (1-t)*(1-b)*b,
                                  (1-b)*(1-b), (1-t)*(1-b) *
                                  b, (t + (1-t)*b)*(1-b),
                                  (t + (1-t)*b)*b, b*(1-b), (t + (1-t)*b)*b])

    single_link_4 = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
    single_link_lik_4 = np.array([(1-b)*(1-b), (1-b)*b, b*(1-t)*(1-b),
                                  b*(t + (1-t)*b), (1-b)*(1-t) *
                                  (1-b), (1-b)*(t + (1-t)*b),
                                  (1-b)*(1-b), (1-b)*b, b*(1-t)*(1-b),
                                  b*(t + (1-t)*b), b*(1-b), b*b])

    single_link_5 = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]])
    single_link_lik_5 = np.array([(1-b)*(1-b), (1-b)*b, b*(1-b),
                                  b*b, (1-b)*(1-b), (1-t)*(1-b)*b,
                                  (1-t)*(1-b)*(1-b), (1-t)*(1-b)*b, b*(1-b),
                                  (t + (1-t)*b)*b, (t + (1-t)*b)*(1-b), (t + (1-t)*b)*b])

    single_link_6 = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]])
    single_link_lik_6 = np.array([(1-b)*(1-b), (1-t)*(1-b)*b, b*(1-b),
                                  b*(t + (1-t)*b), (1-b)*(1-b), (1-b)*b,
                                  (1-b)*(1-t)*(1-b), (1-b) *
                                  (t + (1-t)*b), b*(1-b),
                                  b*b, b*(1-t)*(1-b), b*(t + (1-t)*b)])

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

    likelihoods = [common_cause_1_lik, common_cause_2_lik, common_cause_3_lik,
                   common_effect_1_lik, common_effect_2_lik, common_effect_3_lik,
                   causal_chain_1_lik, causal_chain_2_lik, causal_chain_3_lik,
                   causal_chain_4_lik, causal_chain_5_lik, causal_chain_6_lik,
                   single_link_lik_1, single_link_lik_2, single_link_lik_3,
                   single_link_lik_4, single_link_lik_5, single_link_lik_6]

    hyp_space = {graph_names: DirectedGraph(graph, likelihood, t, b)
                 for (graph_names, graph, likelihood) in
                 zip(graph_names, graphs, likelihoods)}

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

    problem_1_graphs = [hyp_space[hyps] for hyps in
                        ['common_cause_2', 'common_cause_1']]
    problem_2_graphs = [hyp_space[hyps] for hyps in
                        ['common_cause_2', 'common_effect_2']]
    problem_3_graphs = [hyp_space[hyps] for hyps in
                        ['common_cause_2', 'causal_chain_1']]
    problem_4_graphs = [hyp_space[hyps] for hyps in
                        ['common_cause_2', 'causal_chain_4']]
    problem_5_graphs = [hyp_space[hyps] for hyps in
                        ['common_cause_2', 'causal_chain_2']]
    problem_6_graphs = [hyp_space[hyps] for hyps in
                        ['common_cause_2', 'single_link_1']]
    problem_7_graphs = [hyp_space[hyps] for hyps in
                        ['common_cause_2', 'single_link_2']]
    problem_8_graphs = [hyp_space[hyps] for hyps in
                        ['common_cause_2', 'single_link_3']]
    problem_9_graphs = [hyp_space[hyps] for hyps in
                        ['common_cause_1', 'common_effect_2']]
    problem_10_graphs = [hyp_space[hyps] for hyps in
                         ['common_effect_2', 'common_effect_3']]
    problem_11_graphs = [hyp_space[hyps] for hyps in
                         ['common_effect_2', 'causal_chain_1']]
    problem_12_graphs = [hyp_space[hyps] for hyps in
                         ['common_effect_2', 'causal_chain_4']]
    problem_13_graphs = [hyp_space[hyps] for hyps in
                         ['common_effect_2', 'causal_chain_2']]
    problem_14_graphs = [hyp_space[hyps] for hyps in
                         ['common_effect_2', 'single_link_1']]
    problem_15_graphs = [hyp_space[hyps] for hyps in
                         ['common_effect_2', 'single_link_2']]
    problem_16_graphs = [hyp_space[hyps] for hyps in
                         ['common_effect_2', 'single_link_3']]
    problem_17_graphs = [hyp_space[hyps] for hyps in
                         ['causal_chain_1', 'causal_chain_3']]
    problem_18_graphs = [hyp_space[hyps] for hyps in
                         ['causal_chain_1', 'causal_chain_4']]
    problem_19_graphs = [hyp_space[hyps] for hyps in
                         ['causal_chain_1', 'causal_chain_6']]
    problem_20_graphs = [hyp_space[hyps] for hyps in
                         ['causal_chain_1', 'causal_chain_5']]
    problem_21_graphs = [hyp_space[hyps] for hyps in
                         ['causal_chain_1', 'causal_chain_2']]
    problem_22_graphs = [hyp_space[hyps] for hyps in
                         ['causal_chain_1', 'single_link_1']]
    problem_23_graphs = [hyp_space[hyps] for hyps in
                         ['causal_chain_1', 'single_link_2']]
    problem_24_graphs = [hyp_space[hyps] for hyps in
                         ['causal_chain_1', 'single_link_3']]
    problem_25_graphs = [hyp_space[hyps] for hyps in
                         ['causal_chain_1', 'single_link_4']]
    problem_26_graphs = [hyp_space[hyps] for hyps in
                         ['causal_chain_1', 'single_link_5']]
    problem_27_graphs = [hyp_space[hyps] for hyps in
                         ['causal_chain_1', 'single_link_6']]

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


if __name__ == "__main__":
    t = 0.8
    b = 0.0
    hyp_space = create_graph_hyp_space(t=t, b=b)

    # check likelihoods sum to 3.0
    for graph_name, graph in hyp_space.items():
        print(graph.lik)
