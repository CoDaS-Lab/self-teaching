import numpy as np
import matplotlib.pyplot as plt
import ternary
from causal_learning import dag
from causal_learning import utils
from causal_learning.graph_teacher import GraphTeacher


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
        # p(d, i)
        teaching_prior = 1 / (self.n_observations) * \
            np.ones((self.n_hyp, self.n_observations))

        # print(np.sum(teaching_prior))

        # p(d, i|h) \propto p(h|d, i) * p(d, i)
        int_obs_posterior = self.learner_posterior * teaching_prior
        int_obs_posterior = np.divide(int_obs_posterior.T, np.sum(
            int_obs_posterior, axis=1)).T

        # p(h'|h)
        self_teaching_hyp_prior = 1 / self.n_hyp * \
            np.ones((self.n_hyp, self.n_observations))

        # p(d, i, h') = p(d, i| h') * p(h')
        joint_self_teaching_posterior = int_obs_posterior * self_teaching_hyp_prior

        self_teaching_posterior_original = np.sum(
            joint_self_teaching_posterior, axis=0)
        self_teaching_posterior_original = [np.sum(
            self_teaching_posterior_original[self.interventions == i])
            for i in range(self.n_interventions)]

        return self_teaching_posterior_original

    def update_teacher_posterior(self):
        teacher_posterior = np.zeros((self.n_hyp,
                                      self.n_observations))

        lik = self.likelihood()

        for i in range(self.n_interventions):
            denom = np.sum(lik[:, self.interventions == i] *
                           self.learner_prior[:, self.interventions == i],
                           axis=0)

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


if __name__ == "__main__":
    # common cause example
    t = 0.8  # transmission rate
    b = 0.0  # background rate

    # get predictions of self-teaching model for all 27 problems
    active_learning_problems = utils.create_active_learning_hyp_space(t=t, b=b)
    self_teaching_model_predictions = []

    for i, active_learning_problem in enumerate(active_learning_problems):
        gst = GraphSelfTeacher(active_learning_problem)
        gst.update_learner_posterior()
        self_teaching_posterior = gst.update_self_teaching_posterior()
        self_teaching_model_predictions.append(self_teaching_posterior)
        # print("Problem {}:".format(i+1), self_teaching_posterior)

    figure, ax = plt.subplots()
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])

    for i in range(len(self_teaching_model_predictions)):
        # make ternary plot
        scale = 1
        ax = figure.add_subplot(3, 9, i+1)
        tax = ternary.TernaryAxesSubplot(ax=ax)
        figure.set_size_inches(10, 10)
        tax.set_title("Problem {}".format(i+1), fontsize=10)
        tax.boundary(linewidth=2.0)
        # tau_samples = np.random.normal(tau, 0.1, 1000)
        # ig_samples = [np.exp(ig_model_predictions[i]/tau_sample) /
        #               np.sum(np.exp(ig_model_predictions[i]/tau_sample))
        #               for tau_sample in tau_samples]

        tax.scatter([self_teaching_model_predictions[i]],
                    marker='o', color='red')
        tax.clear_matplotlib_ticks()
        ax.set_frame_on(False)

    figure.suptitle("Predictions from the Self-Teaching Model")

    plt.show()
