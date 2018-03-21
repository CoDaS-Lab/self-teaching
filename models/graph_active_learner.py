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
        lik = [h.likelihood() for h in self.hyp]
        return lik
    
    def update_posterior(self):
        """Calculates the posterior over all possible action/outcome pairs
        for each graph"""
        post = self.prior * self.likelihood()
        self.posterior = np.nan_to_num(post / np.sum(post, axis=0))
        
    def prior_entropy(self):
        pass
        
    def posterior_entropy(self):
        return np.nansum(self.posterior * np.log2(1 / self.posterior), axis=0)

if __name__ == "__main__":
    hyp_space = create_graph_hyp_space()
    gal = GraphActiveLearner(hyp_space)
    gal.update_posterior()
    print(np.sum(gal.posterior, axis=0))
