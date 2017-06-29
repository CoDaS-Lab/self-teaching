import numpy as np
import matplotlib.pyplot as plt


class ActiveLearner:
    def __init__(self, num_features):
        assert(num_features > 0)

        self.d = []  # observed data points
        self.num_obs = 0  # number of observed data points
        self.m = 2  # number of possible y values
        self.num_features = num_features
        self.hyp_space = self.create_hyp_space(self.num_features)
        self.num_hyp = len(self.hyp_space)
        self.prior = np.array([1 / self.num_hyp
                               for _ in range(self.num_hyp)])
        self.posterior = self.prior
        self.true_hyp_idx = np.random.randint(len(self.hyp_space))
        self.true_hyp = self.hyp_space[self.true_hyp_idx]

    def create_hyp_space(self, num_features):
        """Creates a hypothesis space of specified size"""

        assert num_features > 0

        hyp_space = []
        for i in range(1, num_features + 1):
            for j in range(num_features - i + 1):
                hyp = [0 for _ in range(num_features)]
                hyp[j:j + i] = [1 for _ in range(i)]
                hyp_space.append(hyp)
        hyp_space = np.array(hyp_space)
        return hyp_space

    def likelihood(self, x, y):
        """Calculates the likelihood of observing the datapoint x"""

        assert y == 0 or y == 1

        lik = np.zeros(len(self.hyp_space))
        for i, hyp in enumerate(self.hyp_space):
            if hyp[x] == y:
                lik[i] = 1
            else:
                lik[i] = 0
        return lik

    def observe(self, x, y):
        """Calculate the posterior based on observing x"""

        assert y == 0 or y == 1

        lik = self.likelihood(x, y)
        posterior = np.array(self.posterior) * np.array(lik)
        if np.sum(posterior) != 0:
            return posterior / np.sum(posterior)
        else:
            return posterior

    def update(self, x, y):
        """Performs Bayesian inference to update the model based on observing x"""

        assert y == 0 or y == 1

        lik = self.likelihood(x, y)
        self.posterior = self.observe(x, y)

    def entropy(self, p):
        """Calculate the entropy of a random variable"""

        # print(np.sum(p))
        # assert np.sum(p) == 1.0  # checks for valid pmf

        p = p[np.nonzero(p)]  # remove all zero probability hypotheses
        entropy = -1 * np.sum(np.log(p) * p)
        return entropy

    def information_gain(self, x, y):
        """Calulate the amount of information gain from a single observation"""
        entropy_prior = self.entropy(self.posterior)
        posterior_new = self.observe(x, y)
        entropy_post = self.entropy(posterior_new)
        information_gain = entropy_prior - entropy_post
        return information_gain

    def expected_information_gain(self, x):
        """Calculate the expected information gain across all possible outcomes"""
        eig_vec = np.zeros(self.m)
        for i, y in enumerate(range(self.m)):
            eig_vec[i] = self.information_gain(x, y)

        # print("eigvec", eig_vec)
        # print("eigvec mean", np.mean(eig_vec))
        return np.mean(eig_vec)

    def run(self):
        """Runs the active learner until the true hypothesis is discovered"""

        # assert self.true_hyp in self.hyp_space

        queries = np.arange(self.num_features)

        # while np.nonzero(self.posterior)[0].shape[0] > 1:
        while np.count_nonzero(self.posterior) > 1:
            eig = np.zeros_like(queries, dtype=np.float)
            for i, query in enumerate(queries):
                eig[i] = self.expected_information_gain(query)

            # select query with maximum expected information gain
            query = queries[np.random.choice(np.where(eig == np.amax(eig))[0])]

            # update model
            query_y = self.true_hyp[query]
            self.update(query, query_y)

            # remove query from set of queries
            query_idx = np.argwhere(queries == query)
            queries = np.delete(queries, query_idx)

            self.num_obs += 1

        # TODO: return actual posterior to check with true hyp
        return self.num_obs


if __name__ == "__main__":
    num_features = 8
    n_iters = 100
    num_obs_sum = 0

    for i in range(n_iters):
        active_learner = ActiveLearner(num_features)
        num_obs = active_learner.run()
        num_obs_sum += num_obs

    print(num_obs_sum / n_iters)
