import numpy as np


class Teacher:
    def __init__(self, num_features):
        self.n = 0
        self.m = 2
        self.num_features = num_features
        self.num_labels = 2
        self.features = np.arange(self.num_features)
        self.labels = np.arange(self.num_labels)
        self.hyp_space = self.create_hyp_space(self.num_features)
        self.num_hyp = len(self.hyp_space)
        self.learner_prior = np.array([[[1 / self.num_hyp
                                         for _ in range(self.num_labels)]
                                        for _ in range(self.num_features)]
                                       for _ in range(self.num_hyp)])
        self.teacher_likelihood = np.array([[[1 / self.num_hyp
                                              for _ in range(self.num_labels)]
                                             for _ in range(self.num_features)]
                                            for _ in range(self.num_hyp)])
        self.learner_posterior = np.array([[[1 / self.num_hyp
                                             for _ in range(self.num_labels)]
                                            for _ in range(self.num_features)]
                                           for _ in range(self.num_hyp)])

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

        lik = np.zeros((self.num_hyp, self.num_features, self.num_labels))

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

    def get_teacher_likelihood(self):
        return self.teacher_likelihood

    def update_learner_posterior(self):
        """Calculates the unnormalized posterior across all 
        possible feature/label observations"""

        lik = self.likelihood()  # p(y|x, h)
        teacher_likelihood = self.get_teacher_likelihood()
        # teacher_likelihood = self.selection_likelihood_two()
        # teacher_likelihood = self.selection_likelihood()  # p_t(x|h)

        # calculate posterior and normalize
        self.learner_posterior = lik * teacher_likelihood * self.learner_prior
        self.learner_posterior = self.learner_posterior / \
            np.sum(self.learner_posterior, axis=0)

    def update_teacher_likelihood(self):
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

        self.teacher_likelihood = prob_conditional_features

    def cooperative_inference(self, n_iters):
        """Run selection and updating equations until convergence"""
        for i in range(n_iters):
            self.update_learner_posterior()
            self.update_teacher_likelihood()


if __name__ == "__main__":
    num_features = 4
    teach = Teacher(num_features)
    teach.update_learner_posterior()
    teach.update_teacher_likelihood()
    print(teach.get_teacher_likelihood())
    teach.update_learner_posterior()
    teach.update_teacher_likelihood()
    print(teach.get_teacher_likelihood())
    teach.update_learner_posterior()
    teach.update_teacher_likelihood()
    print(teach.get_teacher_likelihood())
