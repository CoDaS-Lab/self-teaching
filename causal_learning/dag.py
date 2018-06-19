from collections import deque
import numpy as np


class DirectedGraph:
    def __init__(self, edges, lik=None, transmission_rate=0.9, background_rate=0.05):
        self.adjacency_matrix = edges
        self.n = self.adjacency_matrix.shape[0]
        self.n_actions = self.n
        self.n_observations = 2 ** self.n
        self.transmission_rate = transmission_rate
        self.background_rate = background_rate

        self.observations = np.array([[0, 1, 1], [0, 1, 2],
                                      [0, 2, 1], [0, 2, 2],
                                      [1, 0, 1], [1, 0, 2],
                                      [1, 1, 0], [1, 2, 0],
                                      [2, 0, 1], [2, 0, 2],
                                      [2, 1, 0], [2, 2, 0]])

        if lik is None:
            self.lik = np.zeros(len(self.observations))
        else:
            self.lik = lik

        assert self.n >= 0
        assert self.transmission_rate >= 0.0
        assert self.background_rate >= 0.0

    def get_parents(self, node):
        """Calculate the parents of a given node"""
        return np.flatnonzero(self.adjacency_matrix[:, node])

    def get_children(self, node):
        """Calculate the children of a given node"""
        return np.flatnonzero(self.adjacency_matrix[node])

    # def intervene(self, node):
    #     """Calculate the outcome from intervening on a particular node"""

    #     outcomes = np.zeros(self.n)  # array to store outcomes
    #     outcomes[node] = 1.0  # set intervened node to be on

    #     # temporarily remove edge between node and parent
    #     intervened_parents = self.get_parents(node)
    #     self.adjacency_matrix[intervened_parents, node] = 0

    #     q = deque()  # create queue to store nodes

    #     # set root nodes to be bg rate and add to queue
    #     # root notes have no parents and are not the node intervened on
    #     for i in range(self.n):
    #         if len(self.get_parents(i)) == 0 and i != node:
    #             outcomes[i] = self.background_rate
    #             q.append(i)

    #     # append children of intervened node to queue
    #     children = self.get_children(node)
    #     for child in children:
    #         q.append(child)  # append children of nodes to queue

    #     while len(q) is not 0:
    #         curr_node = q.popleft()  # remove first node from queue

    #         # calculate probability of turning on
    #         parents = self.get_parents(curr_node)
    #         outcomes[curr_node] = np.sum(outcomes[parents]) * \
    #             self.transmission_rate + self.background_rate

    #         # append children to queue
    #         children = self.get_children(curr_node)
    #         for child in children:
    #             q.append(child)

    #     # add edges back in from parents to intervened node
    #     self.adjacency_matrix[intervened_parents, node] = 1

    #     # set any outcomes greater than 1 to 1
    #     outcomes[outcomes > 1.0] = 1.0

    #     return outcomes

    # def likelihood(self):
    #     """Calculate the likelihood of a node being turned on?"""
    #     outcomes = np.zeros((self.n, self.n))
    #     for i in range(self.n):
    #         outcomes[i] = self.intervene(i)

    #     # use outcomes matrix to determine likelihood
    #     lik = np.zeros((self.n_observations, self.n_actions))
    #     for i in range(self.n_observations):
    #         for j in range(self.n_actions):
    #             observation = self.observations[i]
    #             outcome = outcomes[j]
    #             new_outcome = np.zeros_like(outcome)

    #             for k, o in enumerate(observation):
    #                 if np.isclose(o, 0):
    #                     new_outcome[k] = 1 - outcome[k]
    #                 else:
    #                     new_outcome[k] = outcome[k]

    #             lik[i, j] = np.prod(new_outcome)

    #     # check likelihoods for each action sum to 1
    #     np.all(np.sum(lik, axis=0) == 1)

    #     # flatten likelihood into a single dimension
    #     lik = np.array([
    #         lik[4, 0], lik[5, 0], lik[6, 0], lik[7, 0],
    #         lik[2, 1], lik[3, 1], lik[1, 2], lik[3, 2],
    #         lik[6, 1], lik[7, 1], lik[5, 2], lik[7, 2]
    #     ])

    #     return lik
