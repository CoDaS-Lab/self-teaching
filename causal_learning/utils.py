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


def create_graph_hyp_space(t=0.9, b=0.05):
    # enumerate all graphs with three nodes
    common_cause_1 = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]])
    common_cause_1_lik = np.array(
        [((1-t)*(1-b))**2, (1-t)*(1-b)*(t + (1-t)*b), (t + (1-t)*b)*(1-t)*(1-b), (t + (1-t)*b)**2,
         (1-b)**2, (1-b)*b, (1-b)**2, (1-b)*b,
         b*(1-t)*(1-b), b*(t + (1-t)*b), b*(1-t)*(1-b), b*(t + (1-t)*b)])

    common_cause_2 = np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0]])
    common_cause_2_lik = permute_likelihood(common_cause_1_lik, (2, 1, 3))
    common_cause_2_lik[7], common_cause_2_lik[10] = common_cause_2_lik[10], common_cause_2_lik[7]

    common_cause_3 = np.array([[0, 0, 0], [0, 0, 0], [1, 1, 0]])
    common_cause_3_lik = permute_likelihood(common_cause_1_lik, (3, 1, 2))
    common_cause_3_lik[1], common_cause_3_lik[2] = common_cause_3_lik[2], common_cause_3_lik[1]
    common_cause_3_lik[5], common_cause_3_lik[8] = common_cause_3_lik[8], common_cause_3_lik[5]

    common_effect_1 = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]])
    common_effect_1_lik = np.array(
        [(1-b)*(1-t)*(1-b), (1-b)*(t + (1-t)*b), b*(1-t)*(1-t)*(1-b),
         b*(t*(1-t) + (1-t)*t + t*t + b*((1-t)**2)),
         (1-b)*(1-t)*(1-b), (1-b)*(t + (1-t)*b), (1-b)*(1-b), (1-b)*b,
         b*(1-t)*(1-t)*(1-b), b*(t*(1-t) + (1-t)*t + t*t + b*((1-t)**2)), (1-b)*b, b*b])

    common_effect_2 = np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]])
    common_effect_2_lik = permute_likelihood(common_effect_1_lik, (3, 1, 2))
    common_effect_2_lik[1], common_effect_2_lik[2] = common_effect_2_lik[2], common_effect_2_lik[1]

    common_effect_3 = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]])
    common_effect_3_lik = permute_likelihood(common_effect_1_lik, (2, 3, 1))
    common_effect_3_lik[5], common_effect_3_lik[8] = common_effect_3_lik[8], common_effect_3_lik[5]
    common_effect_3_lik[7], common_effect_3_lik[10] = common_effect_3_lik[10], common_effect_3_lik[7]

    causal_chain_1 = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    causal_chain_1_lik = np.array(
        [(1-t)*(1-b)*(1-b), (1-t)*(1-b)*b, (t + (1-t)*b)*(1-t)*(1-b), (t + (1-t)*b)**2,
         (1-b)*(1-t)*(1-b), (1-b)*(t + (1-t)*b), (1-b)**2, (1-b)*b,
         b*(1 - t)*(1-b), b*(t + (1-t)*b), b*(1-t)*(1-b), b*(t + (1-t)*b)]
    )

    causal_chain_2 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    causal_chain_2_lik = permute_likelihood(causal_chain_1_lik, (1, 3, 2))
    causal_chain_2_lik[1], causal_chain_2_lik[2] = causal_chain_2_lik[2], causal_chain_2_lik[1]

    causal_chain_3 = np.array([[0, 0, 1], [1, 0, 0], [0, 0, 0]])
    causal_chain_3_lik = permute_likelihood(causal_chain_1_lik, (2, 3, 1))
    causal_chain_3_lik[5], causal_chain_3_lik[8] = causal_chain_3_lik[8], causal_chain_3_lik[5]
    causal_chain_3_lik[7], causal_chain_3_lik[10] = causal_chain_3_lik[10], causal_chain_3_lik[7]

    causal_chain_4 = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0]])
    causal_chain_4_lik = permute_likelihood(causal_chain_1_lik, (2, 1, 3))
    causal_chain_4_lik[7], causal_chain_4_lik[10] = causal_chain_4_lik[10], causal_chain_4_lik[7]

    causal_chain_5 = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0]])
    causal_chain_5_lik = permute_likelihood(causal_chain_1_lik, (3, 1, 2))
    causal_chain_5_lik[1], causal_chain_5_lik[2] = causal_chain_5_lik[2], causal_chain_5_lik[1]
    causal_chain_5_lik[5], causal_chain_5_lik[8] = causal_chain_5_lik[8], causal_chain_5_lik[5]

    # NOTE: hack! this specifies the likelihoods for a connected chain
    causal_chain_6 = np.array([[0, 0, 1], [0, 0, 0], [0, 1, 0]])
    causal_chain_6_lik = permute_likelihood(causal_chain_1_lik, (3, 2, 1))
    causal_chain_6_lik[1], causal_chain_6_lik[2] = causal_chain_6_lik[2], causal_chain_6_lik[1]
    causal_chain_6_lik[5], causal_chain_6_lik[8] = causal_chain_6_lik[8], causal_chain_6_lik[5]
    causal_chain_6_lik[7], causal_chain_6_lik[10] = causal_chain_6_lik[10], causal_chain_6_lik[7]

    graphs = [common_cause_1, common_cause_2, common_cause_3,
              common_effect_1, common_effect_2, common_effect_3,
              causal_chain_1, causal_chain_2, causal_chain_3,
              causal_chain_4, causal_chain_5, causal_chain_6]

    likelihoods = [common_cause_1_lik, common_cause_2_lik, common_cause_3_lik,
                   common_effect_1_lik, common_effect_2_lik, common_effect_3_lik,
                   causal_chain_1_lik, causal_chain_2_lik, causal_chain_3_lik,
                   causal_chain_4_lik, causal_chain_5_lik, causal_chain_6_lik]

    hyp_space = [DirectedGraph(graph, likelihood, t, b)
                 for (graph, likelihood) in zip(graphs, likelihoods)]
    return hyp_space
