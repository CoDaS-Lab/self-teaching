import numpy as np
import model.utils as utils


class RandomLearner():
    def __init__(self, n_features, hyp_space_type, true_hyp=None):
        self.n_features = n_features
        self.n_labels = 2
        self.observed_features = np.array([])
        self.observed_labels = np.array([])
        self.n_obs = 0
        self.features = np.arange(self.n_features)
        self.labels = np.arange(self.n_labels)
        if hyp_space_type == "boundary":
            self.hyp_space = utils.create_boundary_hyp_space(self.n_features)
        elif hyp_space_type == "line":
            self.hyp_space = utils.create_line_hyp_space(self.n_features)
        self.n_hyp = len(self.hyp_space)
        self.prior = (1 / self.n_hyp) * \
            np.ones((self.n_hyp, self.n_features, self.n_labels))
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
        self.first_feature_prob = np.zeros(self.n_features)

    def likelihood(self):
        lik = np.ones((self.n_hyp, self.n_features, self.n_labels))

        for i, hyp in enumerate(self.hyp_space):
            for j, feature in enumerate(self.features):
                for k, label in enumerate(self.labels):
                    if hyp[feature] == label:
                        lik[i, j, k] = 1
                    else:
                        lik[i, j, k] = 0
        return lik

    def update_posterior(self):
        prior = self.posterior  # previous posterior as new prior
        lik = self.likelihood()
        self.posterior = prior * lik
        self.posterior = self.posterior / np.sum(self.posterior, axis=0)
        self.posterior = np.nan_to_num(self.posterior)

    def run(self):
        hypothesis_found = False
        # true_hyp_found_idx = -1

        queries = np.arange(self.n_features)

        while hypothesis_found is not True:
            if self.n_obs == 0:
                self.first_feature_prob = [
                    1 / len(queries) for _ in range(len(queries))]

            # select a query at random
            query_feature = np.random.choice(queries)
            query_label = self.true_hyp[query_feature]

            # update model
            self.update_posterior()

            # get new posterior and broadcast
            # print(self.posterior)
            # print(query_feature)
            # print(int(query_label))

            updated_learner_posterior = self.posterior[:,
                                                       query_feature,
                                                       int(query_label)]
            # print(updated_learner_posterior)
            self.posterior = np.repeat(updated_learner_posterior,
                self.n_labels * self.n_features).reshape(
                    self.n_hyp,
                    self.n_features,
                    self.n_labels)

            # increment number of observations
            self.n_obs += 1

            # save current posterior of true hypothesis
            self.posterior_true_hyp[self.n_obs] = \
                updated_learner_posterior[self.true_hyp_idx]

            # remove query from set of queries
            query_idx = np.argwhere(queries == query_feature)
            queries = np.delete(queries, query_idx)

            # check for any hypothesis with probability one
            if np.any(updated_learner_posterior == 1):
                hypothesis_found = True
                # true_hyp_found_idx = np.where(updated_learner_posterior == 1)

        return self.n_obs, self.posterior_true_hyp, self.first_feature_prob
