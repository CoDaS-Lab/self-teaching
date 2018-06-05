import numpy as np
from dag import DirectedGraph
from utils import create_graph_hyp_space

class GraphActiveLearner:
    def __init__(self, graphs):
        self.n_hyp = len(graphs)
        self.n_actions = 3
        self.n_observations = 8
        self.hyp = graphs
        self.prior = 1 / self.n_hyp * np.ones((self.n_hyp, self.n_observations, self.n_actions ** 2))
        
    def likelihood(self):
        """Returns the likelihood of each action/observation pair for each graph"""
        # lik = np.array([h.likelihood() for h in self.hyp])
        # return lik

        full_lik = np.empty((self.n_hyp, self.n_observations, self.n_actions ** 2))
        
        for i, h in enumerate(self.hyp):
            lik = h.likelihood()

            l = 0
            for j in range(self.n_actions):
                for k in range(self.n_actions):
                    full_lik[i, :, l] = lik[:, j] * lik[:, k]
                    l += 1

        return full_lik

    
    def update_posterior(self):
        """Calculates the posterior over all possible action/observation pairs
        for each graph"""
        post = self.prior * self.likelihood()
        self.posterior = np.nan_to_num(post / np.sum(post, axis=0))
        
    def prior_entropy(self):
        return np.nansum(self.prior * np.log2(1/self.prior), axis=0)
        
    def posterior_entropy(self):
        log_inv_posterior = np.where(self.posterior > 0, np.log2(1/self.posterior), 0)
        return np.nansum(self.posterior * log_inv_posterior, axis=0)

    def observation_likelihood(self):
        return np.sum(self.prior * self.likelihood(), axis=0)
        
    def expected_information_gain(self):
        eig = self.prior_entropy() - np.sum(self.observation_likelihood() * \
                                            self.posterior_entropy(), axis=0)
        return eig
    
# if __name__ == "__main__":
h1 = np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0]])
h2 = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]])
g1 = DirectedGraph(h1, transmission_rate=0.8, background_rate=0.0)
g2 = DirectedGraph(h2, transmission_rate=0.8, background_rate=0.0)
gal1 = GraphActiveLearner([g1, g2])

gal1.update_posterior()
eig = gal1.expected_information_gain()[0]
print(eig / np.sum(eig))

h3 = np.array([[0, 0, 1], [0, 0, 0], [0, 1, 0]])
h4 = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
g3 = DirectedGraph(h3, transmission_rate=0.8, background_rate=0.0)
g4 = DirectedGraph(h4, transmission_rate=0.8, background_rate=0.0)
gal2 = GraphActiveLearner([g3, g4])

gal2.update_posterior()
eig = gal2.expected_information_gain()[0]
print(np.exp(eig/0.37) / np.sum(np.exp(eig/0.37)))

h5 = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
h6 = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]])
g5 = DirectedGraph(h5, transmission_rate=0.8, background_rate=0.0)
g6 = DirectedGraph(h6, transmission_rate=0.8, background_rate=0.0)
gal3 = GraphActiveLearner([g5, g6])

gal3.update_posterior()
eig = gal3.expected_information_gain()[0]
print(eig / np.sum(eig))

graphs = create_graph_hyp_space()
graph_active_learner = GraphActiveLearner(graphs)
print(graph_active_learner.expected_information_gain()[0])
