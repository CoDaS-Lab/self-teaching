import numpy as np
from causal_learning.dag import DirectedGraph


def create_graph_hyp_space(transmission_rate=0.9, background_rate=0.05):
    # enumerate all graphs with three nodes
    common_cause_1 = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]])
    common_cause_2 = np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0]])
    common_cause_3 = np.array([[0, 0, 0], [0, 0, 0], [1, 1, 0]])
    common_effect_1 = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]])
    common_effect_2 = np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]])
    common_effect_3 = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]])
    causal_chain_1 = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    causal_chain_2 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    causal_chain_3 = np.array([[0, 0, 1], [1, 0, 0], [0, 0, 0]])
    causal_chain_4 = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0]])
    causal_chain_5 = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0]])
    causal_chain_6 = np.array([[0, 0, 1], [0, 0, 0], [0, 1, 0]])

    graphs = [common_cause_1, common_cause_2, common_cause_3,
              common_effect_1, common_effect_2, common_effect_3,
              causal_chain_1, causal_chain_2, causal_chain_3,
              causal_chain_4, causal_chain_5, causal_chain_6]

    hyp_space = [DirectedGraph(graph, transmission_rate, background_rate)
                 for graph in graphs]
    return hyp_space
