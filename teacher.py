import numpy as np


class Teacher:
    def __init__(self):
        self.n = 0
        self.m = 2
        self.num_features = 11
        self.num_labels = 2
        self.features = np.arange(self.num_features)
        self.labels = np.arange(self.num_labels)
        self.hyp_space = self.create_hyp_space(self.num_features)
        self.num_hyp = len(self.hyp_space)
        self.teacher_prior = np.array([1 / self.num_hyp
                                       for _ in range(self.num_hyp)])

        self.learner_prior = np.array([1 / self.num_hyp
                                       for _ in range(self.num_hyp)])
        self.teacher_posterior = None
        self.learner_posterior = None

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
        # marginalize over y
        # normalize over x

    def learn_posterior(self):
        """Calculates the unnormalized posterior across all 
        possible feature/label observations"""
        lik = self.likelihood()
        self.learner_posterior = (self.learner_prior *
                                  lik.T).T  # using broadcasting

    def teach_likelihood(self):
        if self.teacher_lik is None:
            # initialize to random
            self.teacher_lik = np.ones(self.num_features)
            self.teacher_lik = self.teacher_lik / np.sum(self.teacher_lik)
        else:
            # calculate teaching likelihood
            # TODO: fill in eq
            return self.teacher_lik

    def learn_likelihood(self):
        return 0

    def teach_posterior_predictive(self):
        """Calculates the posterior predictive over labels for each feature"""

        # for x in self.features:
        #     for y in self.labels:
        #         current_posterior = self.learner_posterior[:, x, y]
        #         current_hyps = [i for i, hyp in enumerate(self.hyp_space)
        #                         if hyp[x] == y]

        # get indices of hyp_space consistent with hyp[x] == y

        # get these from learner_post

        # sum over/hypothesis averaging
        self.posterior_predictive = np.zeros(
            (self.num_features, self.num_labels))
        self.posterior_predictive[:, 0] = np.sum(
            self.hyp_space, axis=0) / len(self.hyp_space)
        self.posterior_predictive[:, 1] = 1 - np.sum(
            self.hyp_space, axis=0) / len(self.hyp_space)

    def teach_selection(self):
        """Returns a probability distribution over the points to select for teaching"""
        assert self.learner_posterior != None
        assert self.posterior_predictive != None

        joint_prob = ((self.learner_posterior *
                       self.posterior_predictive).T * self.teacher_prior).T
        joint_prob_no_labels = np.sum(joint_prob, axis=2)
        marginalization_constant = np.sum(joint_prob, axis=(1, 2))
        self.teacher_selection = (
            joint_prob_no_labels.T / marginalization_constant).T
