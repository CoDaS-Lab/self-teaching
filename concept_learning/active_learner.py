import numpy as np
import concept_learning.utils as utils


class ActiveLearner:
    def __init__(self, n_features, hyp_space_type,
                 true_hyp=None, sampling="max"):
        assert(n_features > 0)

        self.d = []  # observed data points
        self.n_obs = 0  # number of observed data points
        self.n_labels = 2  # number of possible y values
        self.n_features = n_features

        if hyp_space_type == "boundary":
            self.hyp_space = utils.create_boundary_hyp_space(self.n_features)
        elif hyp_space_type == "line":
            self.hyp_space = utils.create_line_hyp_space(self.n_features)

        self.n_hyp = len(self.hyp_space)
        self.prior = np.array([1 / self.n_hyp
                               for _ in range(self.n_hyp)])
        self.posterior = self.prior

        if true_hyp is not None:
            self.true_hyp = true_hyp
            self.true_hyp_idx = \
                np.where([np.all(true_hyp == hyp)
                          for hyp in self.hyp_space])[0]
        else:
            self.true_hyp_idx = np.random.randint(self.n_hyp)
            self.true_hyp = self.hyp_space[self.true_hyp_idx]

        self.posterior_true_hyp = np.ones(self.n_features + 1)
        self.posterior_true_hyp[0] = 1 / self.n_hyp
        self.first_feature_prob = np.zeros(n_features)
        self.sampling = sampling

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
        """Updates the model based on observing x using Bayesian inference"""

        assert y == 0 or y == 1

        # lik = self.likelihood(x, y)
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
        """Calculate the expected information gain across all outcomes"""
        eig_vec = np.zeros(self.n_labels)
        eig_weights = np.zeros(self.n_labels)
        for i, y in enumerate(range(self.n_labels)):
            # calculate information gain
            eig_vec[i] = self.information_gain(x, y)

            # calculate posterior prob consistent with this observation
            eig_idx = np.where(self.hyp_space[:, x] == y)
            eig_weights[i] = np.sum(self.posterior[eig_idx])

        return np.dot(eig_vec, eig_weights)

    def run(self, n_steps=None):
        """Runs the active learner until the true hypothesis is discovered"""

        # set n steps to be the number of features if not None
        if n_steps is None:
            n_steps = self.n_features

        queries = np.arange(self.n_features)

        # while np.nonzero(self.posterior)[0].shape[0] > 1:
        while np.count_nonzero(self.posterior) > 1 and n_steps > 0:
            eig = np.zeros_like(queries, dtype=np.float)
            for i, query in enumerate(queries):
                eig[i] = self.expected_information_gain(query)

            # save prob of selecting features
            if self.n_obs == 0:
                self.first_feature_prob = eig / np.sum(eig)

            query = -1
            # select query with maximum expected information gain
            if self.sampling == "max":
                query = queries[np.random.choice(
                    np.where(eig == np.amax(eig))[0])]
            else:
                # sample proportionally
                query = np.random.choice(queries,
                                         p=np.abs(eig / np.sum(eig)))

            # update model
            query_y = self.true_hyp[query]
            self.update(query, query_y)

            # increment number of observations and decrease number of steps
            self.n_obs += 1
            n_steps -= 1

            # save current posterior of true hypothesis
            self.posterior_true_hyp[self.n_obs] = \
                self.posterior[self.true_hyp_idx]

        return self.n_obs, self.posterior_true_hyp, self.first_feature_prob
