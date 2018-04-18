import numpy as np
from dag import DirectedGraph
from utils import create_graph_hyp_space

class GraphActiveLearner:
    def __init__(self, graphs):
        self.n_hyp = len(graphs)
        self.actions = 3
        self.outcomes = 3
        self.hyp = graphs
        self.prior = 1 / self.n_hyp * np.ones((self.n_hyp, self.actions, self.outcomes))
        
    def likelihood(self):
        """Returns the likelihood of each action/outcome pair for each graph"""
        lik = np.array([h.likelihood() for h in self.hyp])
        return lik
    
    def update_posterior(self):
        """Calculates the posterior over all possible action/outcome pairs
        for each graph"""
        post = self.prior * self.likelihood()
        self.posterior = np.nan_to_num(post / np.sum(post, axis=0))
        
    def prior_entropy(self):
        return np.nansum(self.prior * np.log2(1/self.prior), axis=0)
        
    def posterior_entropy(self):
        log_inv_posterior = np.where(self.posterior > 0, np.log2(1/self.posterior), 0)
        return np.nansum(self.posterior * log_inv_posterior, axis=0)

    def outcome_likelihood(self):
        return np.sum(self.likelihood(), axis=0) / np.sum(self.likelihood(), axis=(0, 1))
        
    def expected_information_gain(self):
        eig = self.prior_entropy() - np.sum(self.outcome_likelihood() * \
                                            self.posterior_entropy(), axis=0)
        return eig
    
if __name__ == "__main__":
    h1 = np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0]])
    h2 = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]])
    g1 = DirectedGraph(h1, transmission_rate=0.8, background_rate=0.0)
    g2 = DirectedGraph(h2, transmission_rate=0.8, background_rate=0.0)
    gal1 = GraphActiveLearner([g1, g2])

    gal1.update_posterior()
    print(gal1.expected_information_gain())

    h3 = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    h4 = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]])
    g3 = DirectedGraph(h1, transmission_rate=0.8, background_rate=0.0)
    g4 = DirectedGraph(h2, transmission_rate=0.8, background_rate=0.0)
    gal2 = GraphActiveLearner([g3, g4])

    gal2.update_posterior()
    print(gal2.expected_information_gain())


