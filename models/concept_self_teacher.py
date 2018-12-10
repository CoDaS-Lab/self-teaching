import numpy as np
import models.utils as utils


class ConceptSelfTeacher:
    def __init__(self, n_features=3, hyp_space_type="boundary"):
        if hyp_space_type == "boundary":
            self.hyp_space = utils.create_boundary_hyp_space(n_features)
        elif hyp_space_type == "line":
            self.hyp_space = utils.create_line_hyp_space(n_features)

        self.n_features = n_features
        self.n_labels = 2
        self.features = np.arange(self.n_features)
        self.labels = np.arange(self.n_labels)
        self.n_hyp = len(self.hyp_space)

        self.learner_prior = (1 / self.n_hyp) * \
            np.ones((self.n_hyp, self.n_features, self.n_labels))
        self.self_teaching_posterior = np.zeros(
            (self.n_hyp, self.n_features, self.n_labels))

        self.learner_posterior = self.learner_prior

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

    def update_learner_posterior(self):
        """Calculates the unnormalized posterior across all
        possible feature/label observations"""

        lik = self.likelihood()  # p(y|x, h)
        self.learner_posterior = lik * self.learner_posterior
        denom = np.sum(self.learner_posterior, axis=0)

        # normalize across each hypothesis
        self.learner_posterior = np.divide(self.learner_posterior,
                                           denom, where=denom != 0)

        # set learner posterior to zero where denom = 0
        # print(np.isclose(denom, 0.0))
        self.learner_posterior[:, np.isclose(denom, 0)] = 0

    def update_self_teaching_posterior(self):
        """Calculates the posterior of self teaching for determining which points
        to actively select using the teaching equations"""

        # p(x, y)
        teaching_prior = 1 / (self.n_features * self.n_labels) * \
            np.ones((self.n_hyp, self.n_features, self.n_labels))

        # p(g)
        self_teaching_hyp_prior = 1 / self.n_hyp * \
            np.ones((self.n_hyp, self.n_features, self.n_labels))

        # p(g|x, y) * p(x, y) * p(g)
        self_teaching_joint = self.learner_posterior * \
            teaching_prior * self_teaching_hyp_prior

        # Z inverse (\sum_y \sum_x p(g|x, y) p(x, y))
        Z_inv = np.sum(self.learner_posterior * teaching_prior, axis=(1, 2))

        # marginalize over h and y
        self.self_teaching_posterior = np.sum(np.divide(
            self_teaching_joint.T, Z_inv.T, where=Z_inv.T != 0), axis=(0, 2))
