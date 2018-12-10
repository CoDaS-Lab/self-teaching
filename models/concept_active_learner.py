import numpy as np
import models.utils as utils


class ConceptActiveLearner:
    def __init__(self, n_features=3, hyp_space_type="boundary"):
        assert(n_features > 0)

        self.n_features = n_features
        self.n_labels = 2  # number of possible y values

        if hyp_space_type == "boundary":
            self.hyp_space = utils.create_boundary_hyp_space(self.n_features)
        elif hyp_space_type == "line":
            self.hyp_space = utils.create_line_hyp_space(self.n_features)

        self.n_hyp = len(self.hyp_space)

        self.prior = 1 / self.n_hyp * \
            np.ones((self.n_hyp, self.n_features, self.n_labels))

    def likelihood(self):
        """Calculates the likelihood for all possible data points and hypotheses"""

        lik = np.zeros((self.n_hyp, self.n_features, self.n_labels))

        for i, hyp in enumerate(self.hyp_space):
            for j, feature in enumerate(range(self.n_features)):
                for k, label in enumerate(range(self.n_labels)):
                    if hyp[feature] == label:
                        lik[i, j, k] = 1
                    else:
                        lik[i, j, k] = 0

        return lik

    def update_posterior(self):
        """Updates the learner's posterior using Bayes theorem"""

        self.posterior = self.likelihood() * self.prior
        denom = np.sum(self.posterior, axis=0)

        self.posterior = np.nan_to_num(np.divide(
            self.posterior, denom, where=denom != 0))

        # check sum of posterior is either 0s or 1s for each hyp
        # assert np.all(np.logical_or(
        #     np.isclose(np.sum(self.posterior, axis=0), 1.0),
        #     np.isclose(np.sum(self.posterior, axis=0), 0.0)))

    def prior_entropy(self):
        prior_entropy = np.nansum(self.prior * np.log2(1/self.prior), axis=0)

        return prior_entropy

    def posterior_entropy(self):
        inv_posterior = np.divide(1, self.posterior, where=self.posterior != 0)
        log_inv_posterior = np.log2(inv_posterior, where=inv_posterior != 0)
        posterior_entropy = np.nansum(
            self.posterior * log_inv_posterior, axis=0)

        return posterior_entropy

    def observation_likelihood(self):
        obs_lik = np.sum(self.prior * self.likelihood(), axis=0)

        # assert np.array_equal(self.posterior, np.nan_to_num(
        #     np.divide((self.likelihood() * self.prior), obs_lik, where=obs_lik != 0)))

        return obs_lik

    def expected_information_gain(self):
        weighted_posterior_entropy = np.zeros(self.n_features)

        joint_posterior_entropy = self.observation_likelihood() * \
            self.posterior_entropy()

        # sum over possible observations
        weighted_posterior_entropy = np.sum(joint_posterior_entropy, axis=1)

        # take first col of prior entropy
        eig = self.prior_entropy()[:, 0] - weighted_posterior_entropy
        return eig


if __name__ == "__main__":
    hyp_space_type = "boundary"
    n_features = 3

    al = ConceptActiveLearner(n_features, hyp_space_type)
    al.update_posterior()
    active_learning_prob_one = al.expected_information_gain()

    # normalize
    active_learning_prob_one = active_learning_prob_one / \
        np.sum(active_learning_prob_one)

    print(active_learning_prob_one)
