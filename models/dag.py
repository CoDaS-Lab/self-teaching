import numpy as np
from models import utils


class DirectedGraph:
    def __init__(self, edges, cpds, t=0.8, b=0.01):
        self.graph = edges
        self.n_nodes = self.graph.shape[0]
        self.n_actions = self.n_nodes
        self.n_observations = 2 ** self.n_nodes
        self.t = t
        self.b = b

        self.nodes = np.arange(self.n_nodes)
        self.observations = np.array([[0, 1, 1], [0, 1, 2],
                                      [0, 2, 1], [0, 2, 2],
                                      [1, 0, 1], [1, 0, 2],
                                      [1, 1, 0], [1, 2, 0],
                                      [2, 0, 1], [2, 0, 2],
                                      [2, 1, 0], [2, 2, 0]])
        self.cpds = cpds

        assert self.n_nodes >= 0
        assert self.t >= 0.0
        assert self.b >= 0.0

    def get_parents(self, node, graph):
        """Calculate the parents of a given node"""
        return np.flatnonzero(graph[:, node])

    def get_children(self, node, graph):
        """Calculate the children of a given node"""
        return np.flatnonzero(graph[node])

    def intervene(self, intervention):
        """Takes a given observation and intervenes on the node"""

        # check that intervention is valid
        assert intervention >= 0 and intervention < self.n_nodes

        # set intervened node to be one
        self.intervened_observation = np.zeros(self.n_nodes)
        self.intervened_observation[intervention] = 1

        # create intervened adjacency matrix
        self.intervened_graph = self.graph.copy()

        # remove edges from parents to intervened nodes
        intervened_parents = self.get_parents(intervention, self.graph)
        self.intervened_graph[intervened_parents,
                              intervention] = 0

    def observation_likelihood(self, observation):
        """Calculate the likelihood of a given observation"""

        # determine which node to intervene on
        intervened_node = np.where(observation == 0)[0][0]

        # apply intervention
        self.intervene(intervened_node)

        likelihood = 1

        observed_nodes = np.where(observation != 0)[0]

        # subtract one since 1 = off, 2 = on
        self.intervened_observation[observed_nodes] = \
            observation[observed_nodes] - 1

        for node in observed_nodes:
            node_parents = self.get_parents(node, self.intervened_graph)
            observation_idx = np.append(
                self.intervened_observation[node_parents],
                self.intervened_observation[node]).astype(int)
            likelihood = likelihood * \
                self.cpds[node][tuple(observation_idx)]

        return likelihood

    def likelihood(self):
        lik = [self.observation_likelihood(obs) for obs in self.observations]
        return lik


if __name__ == "__main__":
    t = 0.8
    b = 0.01
    hyp_space = utils.create_teaching_hyp_space(t, b)

    for i in range(len(hyp_space)):
        print(i)
        print(hyp_space[i].likelihood())
