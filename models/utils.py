import numpy as np
from models.dag import DirectedGraph

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
        hyp[:i] = [0 for _ in range(i)]
        hyp_space.append(hyp)
    hyp_space = np.array(hyp_space)
    return hyp_space

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
    
    hyp_space = [DirectedGraph(graph, transmission_rate, background_rate) for graph in graphs]
    return hyp_space
