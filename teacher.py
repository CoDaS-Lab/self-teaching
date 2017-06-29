import numpy as np


class Teacher:
    def __init__(self, num_features):
        self.num_features = num_features
        self.num_labels = 2
        self.observed_features = np.array([])
        self.observed_labels = np.array([])
        self.num_obs = 0
        self.features = np.arange(self.num_features)
        self.labels = np.arange(self.num_labels)
        self.hyp_space = self.create_hyp_space(self.num_features)
        self.num_hyp = len(self.hyp_space)
        self.learner_prior = np.array([[[1 / self.num_hyp
                                         for _ in range(self.num_labels)]
                                        for _ in range(self.num_features)]
                                       for _ in range(self.num_hyp)])
        self.teacher_posterior = np.array([[[1 / self.num_hyp
                                             for _ in range(self.num_labels)]
                                            for _ in range(self.num_features)]
                                           for _ in range(self.num_hyp)])
        self.learner_posterior = self.learner_prior
        self.true_hyp_idx = np.random.randint(len(self.hyp_space))
        self.true_hyp = self.hyp_space[self.true_hyp_idx]

    def create_hyp_space(self, num_features):
        hyp_space = []
        for i in range(1, num_features + 1):
            for j in range(num_features - i + 1):
                hyp = [0 for _ in range(num_features)]
                hyp[j:j + i] = [1 for _ in range(i)]
                hyp_space.append(hyp)
        hyp_space = np.array(hyp_space)
        return hyp_space

    def likelihood(self):
        """Calculates the likelihood of observing all possible pairs of data points"""
        # returns a 66 x 11 x 2 matrix

        # TODO: modify function to take in multiple observations

        lik = np.ones((self.num_hyp, self.num_features, self.num_labels))

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

    def get_teacher_posterior(self):
        return self.teacher_posterior

    def set_learner_posterior(self, learner_posterior):
        self.learner_posterior = learner_posterior

    def set_teacher_posterior(self, teacher_posterior):
        self.teacher_posterior = teacher_posterior

    def update_learner_posterior(self):
        """Calculates the unnormalized posterior across all 
        possible feature/label observations"""

        lik = self.likelihood()  # p(y|x, h)
        teacher_posterior = self.get_teacher_posterior()

        # calculate posterior and normalize
        self.learner_posterior = lik * teacher_posterior * \
            self.learner_posterior  # use existing posterior as prior

        self.learner_posterior = self.learner_posterior / \
            np.nansum(self.learner_posterior, axis=0)
        self.learner_posterior = np.nan_to_num(self.learner_posterior)

    def update_teacher_posterior(self):
        """Calculates the likelihood of selecting data points by transforming the 
        posterior p(y|x, h) to p(x|h)"""

        # uniform joint over data, which is broadcasted into correct shape
        prob_joint_data = np.array([[1 / (self.num_features * self.num_labels)
                                     for _ in range(self.num_labels)]
                                    for _ in range(self.num_features)])  # p(x, y)

        prob_joint_data = np.tile(prob_joint_data, (self.num_hyp, 1, 1))

        # multiply with posterior to get overall joint
        # p(h, x, y) = p(h|, x, y) * p(x, y)
        learner_posterior = self.get_learner_posterior()
        prob_joint = learner_posterior * prob_joint_data

        # marginalize over y, i.e. p(h, x), and broadcast result
        prob_joint_hyp_features = np.sum(prob_joint, axis=2)
        prob_joint_hyp_features = np.repeat(
            prob_joint_hyp_features, self.num_labels).reshape(
                self.num_hyp, self.num_features, self.num_labels)

        # divide by prior over hypotheses to get conditional prob
        # p(x|h) = p(h, x)/p(h)
        prob_conditional_features = prob_joint_hyp_features / self.learner_prior

        self.teacher_posterior = prob_conditional_features

    def sample_teacher_posterior(self):
        """Randomly samples a data point based off the teacher's likelihood"""

        # get teacher likelihood and select data point
        teacher_posterior = self.get_teacher_posterior()
        teacher_posterior_true_hyp = teacher_posterior[self.true_hyp_idx, :, 0]

        # select data point, and normalize if possible
        # if np.all(np.sum(teacher_posterior_true_hyp)) != 0:
        teacher_data = np.random.choice(np.arange(self.num_features),
                                        p=teacher_posterior_true_hyp /
                                        np.nansum(teacher_posterior_true_hyp))
        teacher_data = np.nan_to_num(teacher_data)

        return teacher_data

    def cooperative_inference(self, n_iters):
        """Run selection and updating equations until convergence"""
        # TODO: run until convergence
        for i in range(n_iters):
            self.update_learner_posterior()
            self.update_teacher_posterior()

    def run(self):
        """Run teacher until correct hypothesis is determined"""

        # sample a random true hypothesis
        # self.true_hyp_idx = np.random.randint(len(self.hyp_space))
        # self.true_hyp = self.hyp_space[self.true_hyp_idx]

        hypothesis_found = False
        true_hyp_found_idx = -1

        while hypothesis_found != True:
            # run updates for learner posterior and teacher likelihood until convergence
            self.update_learner_posterior()
            self.update_teacher_posterior()

            # sample data point from teacher
            teaching_sample_feature = self.sample_teacher_posterior()
            teaching_sample_label = self.true_hyp[teaching_sample_feature]
            np.append(self.observed_features, teaching_sample_feature)
            np.append(self.observed_labels, teaching_sample_label)
            self.num_obs += 1

            # get learner posterior and broadcast
            updated_learner_posterior = self.learner_posterior[:, teaching_sample_feature,
                                                               teaching_sample_label]

            # update new learner posterior
            self.learner_posterior = np.repeat(updated_learner_posterior, self.num_labels *
                                               self.num_features).reshape(self.num_hyp,
                                                                          self.num_features,
                                                                          self.num_labels)

            # check if any hypothesis has probability one
            if np.any(updated_learner_posterior == 1):
                hypothesis_found = True
                true_hyp_found_idx = np.where(updated_learner_posterior == 1)

        return true_hyp_found_idx, self.num_obs


if __name__ == "__main__":
    num_features = 8
    n_iters = 100
    num_obs_sum = 0

    for i in range(n_iters):
        teacher = Teacher(num_features)
        true_hyp, num_obs = teacher.run()
        num_obs_sum += num_obs

    print(num_obs_sum / n_iters)
