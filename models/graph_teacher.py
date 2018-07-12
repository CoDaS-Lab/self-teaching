import numpy as np
import matplotlib.pyplot as plt
from models import utils


class GraphTeacher:
    def __init__(self, graphs):
        self.n_hyp = len(graphs)
        self.actions = np.array([1, 2, 3])
        self.n_actions = len(self.actions)
        self.observations = np.array([[0, 1, 1], [0, 1, 2],
                                      [0, 2, 1], [0, 2, 2],
                                      [1, 0, 1], [1, 0, 2],
                                      [1, 1, 0], [1, 2, 0],
                                      [2, 0, 1], [2, 0, 2],
                                      [2, 1, 0], [2, 2, 0]])
        self.n_observations = len(self.observations)

        self.interventions = np.array([0, 0, 0, 0,
                                       1, 1, 2, 2,
                                       1, 1, 2, 2])
        self.n_interventions = len(np.unique(self.interventions))
        self.unique_interventions = [3, 9, 11]

        self.hyp = graphs

        # prior over graphs
        self.learner_prior = 1 / self.n_hyp * \
            np.ones((self.n_hyp, self.n_observations))

        # prior over teacher actions
        self.teacher_prior = (1 / self.n_actions) * \
            np.ones((self.n_hyp, self.n_observations))

        # initialize posteriors to be over the priors
        self.learner_posterior = self.learner_prior
        self.teacher_posterior = self.teacher_prior

    def likelihood(self):
        """Calculates p(d|h, i)"""

        self.lik = np.zeros((self.n_hyp,
                             self.n_observations))

        for i, h in enumerate(self.hyp):
            self.lik[:, i] = h.likelihood()

        assert np.isclose(np.sum(self.lik), 36.0)

    def update_teacher_posterior(self, prior):
        """Calculates p(i|h)"""
        teacher_posterior = np.zeros((self.n_hyp, self.n_observations))

        for i in range(self.n_interventions):
            denom = np.sum(self.lik[self.interventions == i] *
                           prior[self.interventions == i], axis=1)

            tmp = ((self.lik[self.interventions == i] *
                    prior[self.interventions == i]).T / denom).T

            tmp = np.sum(tmp, axis=0)
            tmp = tmp / np.sum(tmp)

            teacher_posterior[self.interventions == i, :] = np.tile(
                tmp, (np.sum(self.interventions == i), 1))

        # normalize
        new = teacher_posterior[self.unique_interventions] / \
            np.sum(teacher_posterior[self.unique_interventions], axis=0)

        assert np.isclose(np.sum(new), 12.0)

        teacher_posterior = new[self.interventions]

        # run cooperative inference
        teacher_posterior = self.cooperative_inference(
            teacher_posterior, self.learner_prior)

        return teacher_posterior

    def update_sequential_teacher_posterior(self):
        # calculate updated prior
        self.sequential_prior = np.zeros((self.n_interventions,
                                          self.n_hyp,
                                          self.n_observations))
        self.sequential_teacher_posterior = np.zeros((self.n_interventions,
                                                      self.n_hyp,
                                                      self.n_observations))

        for i in range(self.n_interventions):
            prior_two = np.sum(
                self.learner_posterior[self.interventions == i], axis=0)
            prior_two = prior_two / np.sum(prior_two)
            self.sequential_prior[i] = np.tile(
                prior_two, (self.n_observations, 1))

            self.sequential_teacher_posterior[i] = self.update_teacher_posterior(
                self.sequential_prior[i])
            self.sequential_teacher_posterior[i] = self.cooperative_inference(
                self.sequential_teacher_posterior[i], self.sequential_prior[i])

    def cooperative_inference(self, teacher_posterior, prior):
        """Run cooperative inference"""

        crit = 0.00001
        teacher_posterior_prev = np.zeros_like(teacher_posterior)
        teacher_posterior = teacher_posterior.copy()

        while np.sum(np.absolute(teacher_posterior - teacher_posterior_prev)) > crit:
            teacher_posterior_prev = teacher_posterior.copy()
            teacher_posterior = self.f(teacher_posterior, prior)

            new = teacher_posterior[self.unique_interventions] / \
                np.sum(teacher_posterior[self.unique_interventions], axis=0)
            teacher_posterior = new[self.interventions]

        return teacher_posterior

    def f(self, teacher_posterior, prior):
        for i in range(self.n_interventions):
            denom = np.sum(teacher_posterior[self.interventions == i] *
                           self.lik[self.interventions == i] *
                           prior[self.interventions == i], axis=1)

            tmp = ((teacher_posterior[self.interventions == i] *
                    prior[self.interventions == i]).T / denom).T

            tmp = np.sum(tmp, axis=0)
            tmp = tmp / np.sum(tmp)
            teacher_posterior[self.interventions == i, :] = \
                np.tile(tmp, (np.sum(self.interventions == i), 1))

        teacher_posterior = (teacher_posterior.T /
                             np.sum(teacher_posterior, axis=1)).T
        return teacher_posterior

    def update_learner_posterior(self):
        self.teacher_posterior = self.update_teacher_posterior(
            self.learner_prior)
        posterior = self.lik * self.teacher_posterior * self.learner_prior
        self.learner_posterior = (posterior.T / np.sum(posterior, axis=1)).T
        assert np.allclose(np.sum(self.learner_posterior, axis=1), 1.0)

    def teacher_likelihood(self, likelihood_one, likelihood_two):
        ex_num = [0, 4, 6]
        cause_num = [0, 3, 6]
        teach = np.zeros((len(cause_num), len(ex_num), len(ex_num)))

        for i, cause in enumerate(cause_num):
            for j, ex_one in enumerate(ex_num):
                for k, ex_two in enumerate(ex_num):
                    teach[i, j, k] = likelihood_one[ex_one, cause] * \
                        likelihood_two[j, ex_two, cause]

        return teach

    def plot_teacher_likelihood(self, teach):
        """Visualize the teacher posterior over the three possible kinds of causal graphs"""
        common_cause = [teach[0][0, 0],
                        teach[0][0, 1] + teach[0][1, 0],
                        teach[0][0, 2] + teach[0][2, 0],
                        teach[0][1, 1],
                        teach[0][1, 2] + teach[0][2, 1],
                        teach[0][2, 2]]

        common_effect = [teach[1][0, 0],
                         teach[1][0, 1] + teach[1][1, 0],
                         teach[1][0, 2] + teach[1][2, 0],
                         teach[1][1, 1],
                         teach[1][1, 2] + teach[1][2, 1],
                         teach[1][2, 2]]

        causal_chain = [teach[2][0, 0],
                        teach[2][0, 1] + teach[2][1, 0],
                        teach[2][0, 2] + teach[2][2, 0],
                        teach[2][1, 1],
                        teach[2][1, 2] + teach[2][2, 1],
                        teach[2][2, 2]]

        actions = ['11', '12', '13', '22', '23', '33']
        ind = np.arange(len(actions))

        plt.figure()

        plt.subplot(1, 3, 1)
        plt.bar(ind, common_effect)
        plt.xticks(ind, actions)
        plt.title("Common effect")

        plt.subplot(1, 3, 2)
        plt.bar(ind, causal_chain)
        plt.xticks(ind, actions)
        plt.title("Causal chain")

        plt.subplot(1, 3, 3)
        plt.bar(ind, common_cause)
        plt.xticks(ind, actions)
        plt.title("Common cause")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # run cooperative inference to teach graphs
    graphs = utils.create_teaching_hyp_space(t=0.8, b=0.01)
    graph_teacher = GraphTeacher(graphs)

    graph_teacher.likelihood()
    graph_teacher.update_teacher_posterior(graph_teacher.learner_prior)
    graph_teacher.update_learner_posterior()
    graph_teacher.update_sequential_teacher_posterior()
    teach = graph_teacher.teacher_likelihood(
        graph_teacher.teacher_posterior, graph_teacher.sequential_teacher_posterior)
    print(teach)
    # graph_teacher.plot_teacher_likelihood(teach)
