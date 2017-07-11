import numpy as np
from active_learner import ActiveLearner


class TeacherRollout:
    def __init__(self, n_features, n_steps):
        self.n_features = n_features
        self.n_labels = 2
        self.n_steps = n_steps
        self.observed_features = np.array([])
        self.observed_labels = np.array([])
        self.n_obs = 0
        self.features = np.arange(self.n_features)
        self.labels = np.arange(self.n_labels)
        self.hyp_space = self.create_boundary_hyp_space()
        self.n_hyp = len(self.hyp_space)
        self.learner_prior = np.array([[[1 / self.n_hyp
                                         for _ in range(self.n_labels)]
                                        for _ in range(self.n_features)]
                                       for _ in range(self.n_hyp)])
        self.teacher_prior = np.array([[[1 / self.n_hyp
                                         for _ in range(self.n_labels)]
                                        for _ in range(self.n_features)]
                                       for _ in range(self.n_hyp)])
        self.teaching_posterior = np.array([[[1 / self.n_hyp
                                              for _ in range(self.n_labels)]
                                             for _ in range(self.n_features)]
                                            for _ in range(self.n_hyp)])
        self.learner_posterior = self.learner_prior
        self.true_hyp_idx = np.random.randint(len(self.hyp_space))
        self.true_hyp = self.hyp_space[self.true_hyp_idx]
        self.posterior_true_hyp = np.ones(self.n_features + 1)
        self.posterior_true_hyp[0] = 1 / self.n_hyp

    def create_hyp_space(self):
        """Creates a hypothesis space of line concepts"""
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

    def get_teaching_posterior(self):
        return self.teaching_posterior

    def get_true_hypothesis(self):
        return self.true_hyp

    def set_learner_posterior(self, learner_posterior):
        self.learner_posterior = learner_posterior

    def set_teaching_posterior(self, teaching_posterior):
        self.teaching_posterior = teaching_posterior

    def set_true_hypothesis(self, true_hyp):
        self.true_hyp = true_hyp
        self.true_hyp_idx = np.where(self.true_hyp in self.hyp_space)[0]

    def update_learner_posterior(self):
        """Calculates the unnormalized posterior across all 
        possible feature/label observations"""

        lik = self.likelihood()  # p(y|x, h)
        teaching_posterior = self.get_teaching_posterior()

        # calculate posterior
        self.learner_posterior = lik * teaching_posterior * \
            self.learner_posterior  # use existing posterior as prior

        # normalize across each hypothesis
        self.learner_posterior = np.nan_to_num(self.learner_posterior /
                                               np.sum(self.learner_posterior, axis=0))

    def rollout(self):
        # create active learners and set true hypothesis for each learner
        active_learners = [ActiveLearner(self.n_features)
                           for _ in range(self.n_hyp)]

        # run active learner for n steps
        for i in range(len(active_learners)):
            active_learners[i].set_true_hypothesis(self.hyp_space[i])
            active_learners[i].run(n_steps=self.n_steps)

        # construct matrix of posteriors from each active learner, p(h'|h*)
        transition_prob = np.zeros((self.n_hyp, self.n_hyp))
        for i in range(len(active_learners)):
            transition_prob[i, :] = active_learners[i].posterior

        # calculate p(x|h*) = p(x|h') * p(h'|h*) and marginalize across all hypotheses
        transition_prob = np.broadcast_to(transition_prob, (self.n_labels,
                                                            self.n_features,
                                                            self.n_hyp,
                                                            self.n_hyp)).T

        self.transition_prob = transition_prob

    def update_teaching_posterior(self):
        """Calculates the posterior for self-teaching with rollout"""

        # calculate p(x|h) using the same method as self teaching
        prob_joint_data = np.array([[1 / (self.n_features * self.n_labels)
                                     for _ in range(self.n_labels)]
                                    for _ in range(self.n_features)])  # p(x, y)

        prob_joint_data = np.tile(prob_joint_data, (self.n_hyp, 1, 1))

        # multiply with posterior to get overall joint
        # p(h, x, y) = p(h|, x, y) * p(x, y)
        learner_posterior = self.get_learner_posterior()
        prob_joint = learner_posterior * prob_joint_data

        # marginalize over y, i.e. p(h, x), and broadcast result
        prob_joint_hyp_features = np.sum(prob_joint, axis=2)
        prob_joint_hyp_features = np.repeat(
            prob_joint_hyp_features, self.n_labels).reshape(
                self.n_hyp, self.n_features, self.n_labels)

        # get conditional prob, i.e. p(x|h) = p(h, x) / \sum_x p(h, x)
        prob_conditional_features = prob_joint_hyp_features / \
            np.repeat(np.sum(prob_joint_hyp_features, axis=1),
                      self.n_features).reshape(
                          self.n_hyp, self.n_features, self.n_labels)
        prob_conditional_features = np.nan_to_num(prob_conditional_features)

        # TODO: calculate teaching posterior differently depending on number of observations
        if self.n_obs <= self.n_steps:
            print("using roll out")
            self.teaching_posterior = np.sum(
                prob_conditional_features * self.transition_prob, axis=0)
        else:
            # TODO: check matrix dimensions
            print("using self teaching")
            self.teacher_posterior = np.sum(
                prob_conditional_features * self.learner_posterior, axis=0)

    def sample_teaching_posterior(self):
        """Sample a data point based off the self-teaching posterior"""

        # get teaching posterior and marginalize across all possible hypotheses
        teaching_posterior = self.get_teaching_posterior()
        if self.n_obs <= self.n_steps:
            print("using rollout")
            teaching_posterior = np.sum(
                teaching_posterior * self.transition_prob, axis=(0, 1))
        else:
            # TODO: check matrix dimenisions
            print("using self teaching")
            teaching_posterior = np.sum(
                teaching_posterior * self.learner_posterior, axis=0)

        # normalize and extract first column (columns are duplicates)
        teaching_posterior = teaching_posterior / \
            np.sum(teaching_posterior, axis=0)

        print(teaching_posterior)
        teaching_posterior_sample = teaching_posterior[:, 0]

        # set probability of selecting already observed features to be zero
        self.observed_features = self.observed_features.astype(int)
        if self.observed_features.size != 0:
            teaching_posterior_sample[self.observed_features] = 0

        # select new teaching point proportionally
        if np.all(np.sum(teaching_posterior_sample)) != 0:
            teaching_data = np.random.choice(np.arange(self.n_features),
                                             p=teaching_posterior_sample /
                                             np.nansum(teaching_posterior_sample))
            teaching_data = np.nan_to_num(teaching_data)

        return teaching_data

    def run(self):
        """Run self-teaching with rollout until a correct hypothesis is determined"""

        hypothesis_found = False

        # calculate transition prob at the beginning
        self.rollout()

        while hypothesis_found != True:
            ci_iters = 50
            for i in range(ci_iters):
                self.update_learner_posterior()
                self.update_teaching_posterior

            # sample data point from self-teaching
            teaching_sample_feature = self.sample_teaching_posterior()
            teaching_sample_label = self.true_hyp[teaching_sample_feature]
            self.observed_features = np.append(
                self.observed_features, teaching_sample_feature)
            self.observed_labels = np.append(
                self.observed_labels, teaching_sample_label)

            # get learner posteiror and broadcast
            updated_learner_posterior = self.learner_posterior[:, teaching_sample_feature,
                                                               teaching_sample_label]

            # check for valid probability distribution
            assert np.isclose(np.sum(updated_learner_posterior), 1.0)

            # update new learner posterior by broadcasting
            self.learner_posterior = np.repeat(updated_learner_posterior, self.n_labels *
                                               self.n_features).reshape(self.n_hyp,
                                                                        self.n_features,
                                                                        self.n_labels)

            # check if any hypothesis has probability one
            if np.any(updated_learner_posterior == 1.0):
                hypothesis_found = True
                true_hyp_found_idx = np.where(updated_learner_posterior == 1.0)

            # increment observations
            self.n_obs += 1

            # save posterior probability of true hypothesis
            self.posterior_true_hyp[self.n_obs] = updated_learner_posterior[self.true_hyp_idx]

        return self.n_obs, self.posterior_true_hyp
