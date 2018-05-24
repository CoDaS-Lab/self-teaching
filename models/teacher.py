import numpy as np
import models.utils as utils


class Teacher:
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
        self.learner_prior = (1 / self.n_hyp) * \
            np.ones((self.n_hyp, self.n_features, self.n_labels))
        self.teacher_posterior = (1 / self.n_features * self.n_labels) * \
            np.ones((self.n_hyp, self.n_features, self.n_labels))
        if true_hyp is not None:
            self.true_hyp = true_hyp
            self.true_hyp = self.true_hyp.tolist()
            self.true_hyp_idx = \
                np.where([np.all(true_hyp == hyp)
                          for hyp in self.hyp_space])[0][0]
            self.true_hyp_idx = self.true_hyp_idx.astype(int)
        else:
            self.true_hyp_idx = np.random.randint(self.n_hyp)
            self.true_hyp = self.hyp_space[self.true_hyp_idx]

        self.learner_posterior = self.learner_prior
        self.posterior_true_hyp = np.ones(self.n_features + 1)
        self.posterior_true_hyp[0] = 1 / self.n_hyp
        self.first_feature_prob = np.zeros(self.n_features)

    def likelihood(self):
        """Calculates the likelihood of observing 
        all possible pairs of data points"""
        # returns a 66 x 11 x 2 matrix

        # TODO: modify function to take in multiple observations

        lik = np.ones((self.n_hyp, self.n_features, self.n_labels))

        for i, hyp in enumerate(self.hyp_space):
            for j, feature in enumerate(self.features):
                for k, label in enumerate(self.labels):
                    if hyp[feature] == label:
                        lik[i, j, k] = 1
                    else:
                        lik[i, j, k] = 0
        return lik

    def set_learner_posterior(self, learner_posterior):
        self.learner_posterior = learner_posterior

    def set_teacher_posterior(self, teacher_posterior):
        self.teacher_posterior = teacher_posterior

    def update_learner_posterior(self):
        """Calculates the unnormalized posterior across all
        possible feature/label observations"""

        lik = self.likelihood()  # p(y|x, h)
        teacher_posterior = self.teacher_posterior

        # calculate posterior and normalize
        self.learner_posterior = lik * teacher_posterior * \
            self.learner_posterior  # use existing posterior as prior

        self.learner_posterior = self.learner_posterior / \
            np.nansum(self.learner_posterior, axis=0)
        self.learner_posterior = np.nan_to_num(self.learner_posterior)

        # assertion to check all posteriors sum to one
        # print(self.learner_posterior)
        # assert np.allclose(np.sum(self.learner_posterior, axis=0), 1.0)

    def update_teacher_posterior(self):
        """Calculates the posterior of selecting data points by transforming the
        posterior p(y|x, h) to p(x|h)"""

        # uniform joint over data, which is broadcasted into correct shape
        prob_joint_data = 1 / (self.n_features * self.n_labels) * \
            np.ones((self.n_features, self.n_labels))  # p(x, y)
        prob_joint_data = np.tile(prob_joint_data, (self.n_hyp, 1, 1))

        # multiply with posterior to get overall joint
        # p(h, x, y) = p(h|, x, y) * p(x, y)
        learner_posterior = self.learner_posterior
        prob_joint = learner_posterior * prob_joint_data

        # marginalize over y, i.e. p(h, x), and broadcast result
        prob_joint_hyp_features = np.sum(prob_joint, axis=2)
        prob_joint_hyp_features = np.repeat(
            prob_joint_hyp_features, self.n_labels).reshape(
                self.n_hyp, self.n_features, self.n_labels)

        # divide by prior over hypotheses to get conditional prob
        # marginalize over x, i.e. p(x | h) = p(h, x) / \sum_x p(h, x)
        prob_conditional_features = prob_joint_hyp_features / \
            np.repeat(np.sum(prob_joint_hyp_features, axis=1),
                      self.n_features).reshape(
                          self.n_hyp, self.n_features, self.n_labels)

        self.teacher_posterior = np.nan_to_num(prob_conditional_features)

    def sample_teacher_posterior(self):
        """Randomly samples a data point based off the teacher's posterior"""

        # get teacher likelihood and select data point
        teacher_posterior = self.teacher_posterior
        teacher_posterior_true_hyp = teacher_posterior[self.true_hyp_idx, :, 0]

        # set probability of selecting observed features to be zero
        self.observed_features = self.observed_features.astype(int)
        if self.observed_features.size != 0:
            teacher_posterior_true_hyp[self.observed_features] = 0

        if self.n_obs == 0:
            self.first_feature_prob = teacher_posterior_true_hyp

        # select max
        teacher_data = self.features[np.random.choice(
            np.where(teacher_posterior_true_hyp ==
                     np.amax(teacher_posterior_true_hyp))[0])]

        return teacher_data

    def run_ci(self, n_iters=5):
        """Run cooperative inference for n_iters"""
        for i in range(n_iters):
            self.update_learner_posterior()
            self.update_teacher_posterior()

    def run(self):
        """Run teacher until correct hypothesis is determined"""

        hypothesis_found = False

        while hypothesis_found is not True:
            # run ci updates for learner posterior and teacher likelihood
            self.run_ci()

            # sample data point from teacher
            teaching_sample_feature = self.sample_teacher_posterior()
            teaching_sample_label = self.true_hyp[teaching_sample_feature]

            # get learner posterior and broadcast
            updated_learner_posterior = \
                self.learner_posterior[:,
                                       teaching_sample_feature,
                                       teaching_sample_label]

            # update new learner posterior
            self.learner_posterior = np.repeat(
                updated_learner_posterior, self.n_labels * self.n_features).reshape(
                    self.n_hyp, self.n_features, self.n_labels)

            # check if any hypothesis has probability one
            if np.any(updated_learner_posterior == 1) and \
                   self.true_hyp_idx == \
                   np.asscalar((np.where(updated_learner_posterior == 1.0))[0]):

                hypothesis_found = True
                true_hyp_found_idx = np.where(updated_learner_posterior == 1)

            # save observed feature and label
            self.observed_features = np.append(
                self.observed_features, teaching_sample_feature)
            self.observed_labels = np.append(
                self.observed_labels, teaching_sample_label)
                
            # increment observations
            self.n_obs += 1

            # save posterior probability of true hypothesis
            self.posterior_true_hyp[self.n_obs] = updated_learner_posterior[self.true_hyp_idx]

        return self.n_obs, self.posterior_true_hyp, self.first_feature_prob
