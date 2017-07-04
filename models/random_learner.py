import numpy as np


class RandomLearner():
    def __init__(self, n_features):
        self.n_features = n_features
        self.num_labels = 2
        self.observed_features = np.array([])
        self.observed_labels = np.array([])
        self.num_obs = 0
        self.features = np.arange(self.n_features)
        self.labels = np.arange(self.num_labels)
        # self.hyp_space = self.create_hyp_space(self.n_features)
        self.hyp_space = self.create_boundary_hyp_space()
        self.num_hyp = len(self.hyp_space)
        self.prior = np.array([[[1 / self.num_hyp
                                 for _ in range(self.num_labels)]
                                for _ in range(self.n_features)]
                               for _ in range(self.num_hyp)])
        self.posterior = self.prior
        self.true_hyp_idx = np.random.randint(len(self.hyp_space))
        self.true_hyp = self.hyp_space[self.true_hyp_idx]
        self.posterior_true_hyp = np.ones(self.n_features + 1)
        self.posterior_true_hyp[0] = 1 / self.num_hyp

    def create_hyp_space(self, n_features):
        hyp_space = []
        for i in range(1, n_features + 1):
            for j in range(n_features - i + 1):
                hyp = [0 for _ in range(n_features)]
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
        lik = np.ones((self.num_hyp, self.n_features, self.num_labels))

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
        true_hyp_found_idx = -1

        queries = np.arange(self.n_features)

        while hypothesis_found != True:
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
                                                       query_feature, int(query_label)]
            # print(updated_learner_posterior)
            self.posterior = np.repeat(updated_learner_posterior,
                                       self.num_labels * self.n_features).reshape(
                                           self.num_hyp,
                                           self.n_features,
                                           self.num_labels)

            # increment number of observations
            self.num_obs += 1

            # save current posterior of true hypothesis
            self.posterior_true_hyp[self.num_obs] = updated_learner_posterior[self.true_hyp_idx]

            # remove query from set of queries
            query_idx = np.argwhere(queries == query_feature)
            queries = np.delete(queries, query_idx)

            # check for any hypothesis with probability one
            if np.any(updated_learner_posterior == 1):
                hypothesis_found = True
                true_hyp_found_idx = np.where(updated_learner_posterior == 1)

        return self.num_obs, self.posterior_true_hyp


if __name__ == "__main__":
    n_features = 8
    n_iters = 1000
    num_obs_arr = np.array([])

    for i in range(n_iters):
        random_learner = RandomLearner(n_features)
        true_hyp, num_obs = random_learner.run()
        num_obs_arr = np.append(num_obs_arr, num_obs)

    print(np.bincount(num_obs_arr.astype(int)) / n_iters)
