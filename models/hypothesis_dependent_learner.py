import numpy as np


class HypothesisDependentLearner:
    def __init__(self, n_features, hyp_space_type):
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
        self.prior = np.array([[[1 / self.n_hyp
                                 for _ in range(self.n_labels)]
                                for _ in range(self.n_features)]
                               for _ in range(self.n_hyp)])
        self.posterior = self.prior
        self.true_hyp_idx = np.random.randint(len(self.hyp_space))
        self.true_hyp = self.hyp_space[self.true_hyp_idx]
        self.current_hyp_idx = np.random.randint(len(self.hyp_space))
        self.current_hyp = self.hyp_space[self.current_hyp_idx]
        self.posterior_true_hyp = np.zeros(self.n_features + 1)
        self.posterior_true_hyp[0] = 1 / self.n_hyp

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

    def run(self):
        """Run hypothesis dependent sampling learner"""

        hypothesis_found = False

        while not hypothesis_found:
            if self.hyp_space_type == "boundary":
                # select a query that is on the boundary
                # check for two edge cases
                if np.all(self.current_hyp == 1):
                    query_feature = 0  # check first feature
                elif np.all(self.current_hyp == 0):
                    query_feature = self.n_features - 1  # check final feature
                else:
                    # check features on the boundary
                    boundary_idx = np.asscalar(
                        np.where(self.current_hyp == 1)[0][0])
                    query_features = self.features[boundary_idx -
                                                   1: boundary_idx]
                    query_feature = np.random.choice(query_features)
            elif self.hyp_space_type == "line":
                # strong sampling for line hypothesis space
                if np.all(self.current_hyp == 1):
                    # check first or last features
                    query_features = np.array([0, self.n_features - 1])
                    query_feature = np.random.choice(query_features)
                elif np.all(self.current_hyp == 0):
                    # check middle features
                    query_features = np.array(
                        [int(self.n_features / 2) - 1, int(self.n_features)])
                    query_feature = np.random.choice(query_features)
                else:
                    # find boundaries by finding ones
                    hyp_len = np.sum(self.current_hyp)
                    hyp_start = np.asscalar(
                        np.where(self.current_hyp == 1)[0][0])
                    hyp_end = hyp_start + hyp_len - 1
                    query_features = np.array([hyp_start, hyp_end])
                    if hyp_start != 0:
                        query_features = np.append(
                            query_features, hyp_start - 1)
                    if hyp_end != self.n_features - 1:
                        query_features = np.append(query_features, hyp_end + 1)
                    query_feature = np.random.choice(query_features)

            # get true label
            query_label = self.true_hyp[query_feature]
            current_label = self.current_hyp[query_feature]  # remove, not used

            if self.current_hyp_idx == self.true_hyp_idx:
                hypothesis_found = True
                consistent_hyps = [1]
                self.posterior_true_hyp[self.n_obs + 1:] = 1.0
            # check if current hypothesis does not match true label
            else:
                # select a new hypothesis that is consistent with all past observations
                self.observed_features = np.append(
                    self.observed_features, query_feature)
                self.observed_labels = np.append(
                    self.observed_labels, query_label)

                consistent_hyps = np.ones(self.n_hyp)
                for i in range(len(self.observed_labels)):
                    # print(i)
                    # print(consistent_hyps)
                    # print(self.observed_features)
                    # print(self.observed_labels)
                    consistent_hyps = np.logical_and(
                        consistent_hyps,
                        self.hyp_space[:,
                                       int(self.observed_features[i])] ==
                        self.observed_labels[i])

                # select a new hypothesis that is consistent with the observed label
                # consistent_hyps = np.where(
                #     self.hyp_space[:, query_feature] == query_label)[0]
                consistent_hyps = np.where(consistent_hyps)[0]
                # print(consistent_hyps)
                self.current_hyp_idx = np.random.choice(consistent_hyps)
                self.current_hyp = self.hyp_space[self.current_hyp_idx]

            # increment observations
            self.n_obs += 1

            # save 1 / number of consistent obs as current prob?
            if self.n_obs < self.n_features:
                self.posterior_true_hyp[self.n_obs] = 1 / len(consistent_hyps)
            else:
                print("too long")
                break

        return self.n_obs, self.posterior_true_hyp
