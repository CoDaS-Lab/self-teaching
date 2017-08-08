import numpy as np


class BayesianLearner:
    def __init__(self, n_features, hyp_space_type, true_hyp=None):
        self.n_features = n_features
        self.n_labels = 2
        self.observed_features = np.array([])
        self.observed_labels = np.array([])
        self.n_obs = 0
        self.features = np.arange(self.n_features)
        self.labels = np.arange(self.n_labels)
        self.hyp_space_type = hyp_space_type
        if hyp_space_type == "boundary":
            self.hyp_space = self.create_boundary_hyp_space()
        elif hyp_space_type == "line":
            self.hyp_space = self.create_line_hyp_space()
        self.n_hyp = len(self.hyp_space)
        self.learner_prior = np.array([[[1 / self.n_hyp
                                         for _ in range(self.n_labels)]
                                        for _ in range(self.n_features)]
                                       for _ in range(self.n_hyp)])
        self.learner_posterior = self.learner_prior

        if true_hyp is not None:
            self.true_hyp = true_hyp
            self.true_hyp_idx = \
                np.where([np.all(true_hyp == hyp)
                          for hyp in self.hyp_space])[0]
        else:
            self.true_hyp_idx = np.random.randint(len(self.hyp_space))
            self.true_hyp = self.hyp_space[self.true_hyp_idx]

        self.posterior_true_hyp = np.zeros(self.n_features + 1)
        self.posterior_true_hyp[0] = 1 / self.n_hyp
        self.queries = np.arange(self.n_features)
        self.first_feature_prob = np.zeros(self.n_features)

    def create_line_hyp_space(self):
        """Creates a hypothesis space of concepts"""
        hyp_space = []
        for i in range(1, self.n_features + 1):
            for j in range(self.n_features - i + 1):
                hyp = [0 for _ in range(self.n_features)]
                hyp[j:j + i] = [1 for _ in range(i)]
                hyp_space.append(hyp)
        hyp_space = np.array(hyp_space)
        return hyp_space

    def create_boundary_hyp_space(self):
        """Creates a hypothesis space of concepts defined by a linear boundary"""
        hyp_space = []
        for i in range(self.n_features + 1):
            hyp = [1 for _ in range(self.n_features)]
            hyp[:i] = [0 for _ in range(i)]
            hyp_space.append(hyp)
        hyp_space = np.array(hyp_space)
        return hyp_space

    def likelihood(self):
        """Calculates the likelihood of observing all possible pairs of data points"""
        # returns a 66 x 11 x 2 matrix

        lik = np.ones((self.n_hyp, self.n_features, self.n_labels))

        for i, hyp in enumerate(self.hyp_space):
            for j, feature in enumerate(self.features):
                for k, label in enumerate(self.labels):
                    if hyp[feature] == label:
                        lik[i, j, k] = 1
                    else:
                        lik[i, j, k] = 0
        return lik

    def get_learner_posterior(self):
        return self.learner_posterior

    def set_learner_posterior(self, learner_posterior):
        self.learner_posterior = learner_posterior

    def update_learner_posterior(self):
        """Calculates the posterior using Bayes rule"""

        lik = self.likelihood()
        learner_posterior = lik * self.learner_posterior  # prior
        self.learner_posterior = np.nan_to_num(
            learner_posterior / np.sum(learner_posterior, axis=0))

    def run(self):
        """Run learner until correct hypothesis is determined"""

        hypothesis_found = False

        while hypothesis_found != True:
            self.update_learner_posterior()

            if self.n_obs == 0:
                self.first_feature_prob = [
                    1 / len(self.queries) for _ in range(len(self.queries))]

            # select a query at random
            query_feature = np.random.choice(self.queries)
            query_label = self.true_hyp[query_feature]

            if self.hyp_space_type == "boundary":
                if query_label == 0:
                    # remove all queries to the left that are do not provide new information
                    self.queries = np.delete(
                        self.queries, np.where(self.queries < query_feature))
                elif query_label == 1:
                    # remove all queries to the right for the same reason
                    self.queries = np.delete(
                        self.queries, np.where(self.queries > query_feature))

                    # update posterior
            updated_posterior = self.learner_posterior[:,
                                                       query_feature,
                                                       query_label]

            # broadcast
            self.learner_posterior = np.repeat(
                updated_posterior, self.n_labels * self.n_features).reshape(
                    self.n_hyp, self.n_features, self.n_labels)

            # check if hypothesis is found
            if np.any(updated_posterior == 1.0):
                found_hyp_idx = np.asscalar(
                    (np.where(updated_posterior == 1.0))[0])
                if found_hyp_idx == self.true_hyp_idx:
                    hypothesis_found = True

                # set posterior prob to be 1 for all future trials
                self.posterior_true_hyp[self.n_obs + 1:] = 1

            # remove query from set of queries
            query_idx = np.argwhere(self.queries == query_feature)
            self.queries = np.delete(self.queries, query_idx)

            # increment observations
            self.n_obs += 1

            # save posterior probability of true hypothesis
            self.posterior_true_hyp[self.n_obs] = updated_posterior[self.true_hyp_idx]

        return self.n_obs, self.posterior_true_hyp, self.first_feature_prob
