import numpy as np


class HypothesisDependentLearner:
    def __init__(self, n_features):
        self.n_features = n_features
        self.n_labels = 2
        self.observed_features = np.array([])
        self.observed_labels = np.array([])
        self.n_obs = 0
        self.features = np.arange(self.n_features)
        self.labels = np.arange(self.n_labels)
        self.hyp_space = self.create_boundary_hyp_space()
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
        self.posterior_true_hyp = np.ones(self.n_features + 1)
        self.posterior_true_hyp[0] = 1 / self.n_hyp

    def create_hyp_space(self):
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
            # select a query that is on the boundary
            # check for two edge cases
            if np.all(self.current_hyp == 1):
                # print("all ones")
                query_feature = 0  # check first feature
            elif np.all(self.current_hyp == 0):
                # print("all zeros")
                query_feature = self.n_features - 1  # check final feature
            else:
                # print("blah")
                # check features on the boundary
                boundary_idx = np.asscalar(
                    np.where(self.current_hyp == 1)[0][0])
                # print("boundary idx", boundary_idx)
                # print(self.features)
                query_features = self.features[boundary_idx -
                                               1: boundary_idx]
                # print("query features", query_features)
                query_feature = np.random.choice(query_features)

            # get true label
            query_label = self.true_hyp[query_feature]
            current_label = self.current_hyp[query_feature]

            print("current", self.current_hyp_idx)
            print("true", self.true_hyp_idx)

            if self.current_hyp_idx == self.true_hyp_idx:
                print("found hypothesis!")
                hypothesis_found = True
            # check if current hypothesis does not match true label
            else:
                # select a new hypothesis that is consistent with the observed label
                consistent_hyps = np.where(
                    self.hyp_space[:, query_feature] == query_label)[0]
                print("consistent", consistent_hyps)
                self.current_hyp_idx = np.random.choice(consistent_hyps)
                self.current_hyp = self.hyp_space[self.current_hyp_idx]

        return self.current_hyp_idx
