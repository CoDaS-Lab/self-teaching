import numpy as np
from models import dag
from models import utils
from models.graph_teacher import GraphTeacher


class GraphSelfTeacher:
    def __init__(self, graphs):
        self.hyp = graphs
        self.n_hyp = len(graphs)

        self.actions = np.array([1, 2, 3])
        self.n_actions = len(self.actions)

        # the set of possible observations
        # 0 = intervene, 1 = observed_off, 2 = observed_on
        self.observations = np.array([[0, 1, 1], [0, 1, 2],
                                      [0, 2, 1], [0, 2, 2],
                                      [1, 0, 1], [1, 0, 2],
                                      [1, 1, 0], [1, 2, 0],
                                      [2, 0, 1], [2, 0, 2],
                                      [2, 1, 0], [2, 2, 0]])
        self.n_observations = len(self.observations)

        # the set of possible interventions
        self.interventions = np.array([0, 0, 0, 0,
                                       1, 1, 2, 2,
                                       1, 1, 2, 2])
        self.n_interventions = len(np.unique(self.interventions))
        self.unique_interventions = [3, 9, 11]

        # initialize priors and posteriors
        self.learner_prior = 1 / self.n_hyp * \
            np.ones((self.n_hyp, self.n_observations))
        self.self_teaching_prior = 1 / self.n_actions * \
            np.ones((self.n_hyp, self.n_observations))

        self.learner_posterior = np.zeros_like(self.learner_prior)
        self.self_teaching_posterior = np.zeros_like(
            self.self_teaching_prior)

    def likelihood(self):
        """Calculate p(d|h, i)"""

        lik = np.zeros((self.n_hyp,
                        self.n_observations))

        for i, h in enumerate(self.hyp):
            lik[i] = h.likelihood()

        # the likelihood should sum to 3.0
        assert np.allclose(np.sum(lik, axis=1), 3.0)

        return lik

    def update_learner_posterior(self):
        self.learner_posterior = self.likelihood() * \
            self.self_teaching_prior * self.learner_prior
        denom = np.sum(self.learner_posterior, axis=0)

        self.learner_posterior = np.divide(
            self.learner_posterior, denom, where=denom != 0)

        # check posterior is normalized
        assert np.all(np.logical_or(
            np.isclose(np.sum(self.learner_posterior, axis=0), 1.0),
            np.isclose(np.sum(self.learner_posterior, axis=0), 0.0)))

    def update_self_teaching_posterior(self):
        # p(x, y)
        teaching_prior = 1 / (self.n_observations * self.n_interventions) * \
            np.ones((self.n_hyp, self.n_observations))

        # print(np.sum(teaching_prior))

        # p(x, y|h) \propto p(h|x, y) * p(x, y)
        int_obs_posterior = self.learner_posterior * teaching_prior
        # normalize by Z
        int_obs_posterior = np.divide(int_obs_posterior.T, np.sum(
            int_obs_posterior, axis=1)).T

        # p(g)
        self_teaching_hyp_prior = 1 / self.n_hyp * \
            np.ones((self.n_hyp, self.n_observations))

        # p(x, y, g) = p(x, y| g) * p(g)
        joint_self_teaching_posterior = int_obs_posterior * self_teaching_hyp_prior

        # \sum_g p(x, y, g)
        self_teaching_posterior_original = [[np.sum(
            joint_self_teaching_posterior[i][self.interventions == j])
            for j in range(self.n_interventions)] for i in range(self.n_hyp)]
        self_teaching_posterior_original = np.sum(
            self_teaching_posterior_original, axis=0)

        # original code
        self_teaching_posterior_two = np.sum(
            joint_self_teaching_posterior, axis=0)
        print(self_teaching_posterior_two)
        # self_teaching_posterior_original = [np.sum(
        #     self_teaching_posterior_original[self.interventions == i])
        #     for i in range(self.n_interventions)]

        return self_teaching_posterior_original

    def update_teacher_posterior(self):
        # initialize empty posterior
        teacher_posterior = np.zeros((self.n_hyp,
                                      self.n_observations))

        # p(y|x, h)
        lik = self.likelihood()

        for i in range(self.n_interventions):
            # \sum_h' p(y|x, h) * p(h)
            denom = np.sum(lik[:, self.interventions == i] *
                           self.learner_prior[:, self.interventions == i],
                           axis=0)

            # p(y|x, h) * p(h)
            numer = lik[:, self.interventions == i] * \
                self.learner_prior[:, self.interventions == i]

            tmp = np.divide(numer, denom, where=denom != 0.0)

            tmp = np.sum(tmp, axis=1)
            tmp = tmp / np.sum(tmp)

            teacher_posterior[:, self.interventions == i] = np.tile(
                tmp, (np.sum(self.interventions == i), 1)).T

        # normalize
        new = (teacher_posterior[:, self.unique_interventions].T /
               (np.sum(teacher_posterior[:, self.unique_interventions], axis=1))).T

        teacher_posterior = new[:, self.interventions]

        return teacher_posterior
