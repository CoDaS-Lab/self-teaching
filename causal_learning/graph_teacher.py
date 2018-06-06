import numpy as np
import matplotlib.pyplot as plt
from causal_learning.utils import create_graph_hyp_space


class GraphTeacher:
    def __init__(self, graphs):
        self.n_hyp = len(graphs)
        self.actions = np.array([1, 2, 3])
        self.n_actions = len(self.actions)
        self.observations = np.array([[0, 0, 0], [0, 0, 1],
                                      [0, 1, 0], [0, 1, 1],
                                      [1, 0, 0], [1, 0, 1],
                                      [1, 1, 0], [1, 1, 1]])
        self.n_observations = len(self.observations)
        self.hyp = graphs

        # prior over graphs
        self.learner_prior = 1 / self.n_hyp * np.ones((self.n_hyp,
                                                       self.n_observations,
                                                       self.n_actions ** 2))
        # prior over teaching actions
        self.teacher_prior = (1 / self.n_actions ** 2) * \
            np.ones((self.n_hyp,
                     self.n_observations,
                     self.n_actions ** 2))

        # initialize posteriors to be over the priors
        self.learner_posterior = self.learner_prior
        self.teacher_posterior = self.teacher_prior

    def likelihood(self):
        """Returns the likelihood of each action/outcome pair for each graph"""

        full_lik = np.empty((self.n_hyp,
                             self.n_observations,
                             self.n_actions ** 2))

        for i, h in enumerate(self.hyp):
            lik = h.likelihood()

            l = 0
            for j in range(self.n_actions):
                for k in range(self.n_actions):
                    full_lik[i, :, l] = lik[:, j] * lik[:, k]
                    l += 1

        return full_lik

    def update_learner_posterior(self):
        """Calculates the posterior over all possible action/outcome pairs
        for each graph"""

        # TODO: uncomment later to use cooperative inference
        # if self.teacher_posterior.shape == (self.n_hyp, self.n_actions ** 2):
        #     # reshape to add dimension across observations
        #     self.teacher_posterior = np.repeat(
        #         self.teacher_posterior[:, np.newaxis, :],
        #         self.n_observations, axis=1)

        # p(g|a, o) = p(o, a|g) * p(g)
        post = self.learner_posterior * self.likelihood()
        self.learner_posterior = np.nan_to_num(post / np.sum(post, axis=0))

    def update_teacher_posterior(self):
        """Calculates the posterior of selecting which actions to take"""
        joint_action_obs = 1 / (self.n_actions ** 2 * self.n_observations) * \
            np.ones((self.n_hyp,
                     self.n_observations,
                     self.n_actions ** 2))  # p(a, o)

        # p(g, a, o) = p(g|a, o) * p(a, o)
        joint_all = self.learner_posterior * joint_action_obs

        # p(g, a) = \sum_o p(g, a, o)
        joint_actions = np.sum(joint_all, axis=1)

        # p(a|g) = p(g, a) / p(g) = p(g, a) / \sum_a p(g, a)
        self.teacher_posterior = (joint_actions.T /
                                  (np.sum(joint_actions, axis=1)).T).T

    def run_cooperative_inference(self, n_iters=1):
        """Run cooperative inference for n_iters"""
        for i in range(n_iters):
            self.update_learner_posterior()
            self.update_teacher_posterior()

    def combine_actions(self, action_posterior):
        """Combine posterior of pairs of actions that are the same"""
        new_action_post = [action_posterior[0],
                           action_posterior[1] + action_posterior[3],
                           action_posterior[2] + action_posterior[6],
                           action_posterior[4],
                           action_posterior[5] + action_posterior[7],
                           action_posterior[8]]
        return new_action_post

    def marginalize_learner_posterior(self):
        """Marginalizes out observations in the learner's posterior"""

    def plot_learner_posterior_by_actions(self):
        """Visualize the marginalized learner posterior distribution over actions"""
        self.action_learner_posterior = np.sum(self.learner_posterior, axis=1)

        # assert np.allclose(np.sum(graph_teacher.action_learner_posterior, axis=0), 1.0)

        actions = ['11', '12', '13', '21', '22', '23', '31', '23', '33']
        hypotheses = ['CC1', 'CC2', 'CC3', 'CE1', 'CE2', 'CE3',
                      'CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH6']
        ind = np.arange(len(hypotheses))
        
        plt.figure()

        for i in range(len(actions)):
            plt.subplot(3, 3, i+1)
            plt.bar(ind, self.action_learner_posterior[:, i])
            plt.xticks(ind, hypotheses)
            plt.title(actions[i])

        plt.tight_layout()
        plt.show()

    def plot_learner_posterior_by_hypotheses(self):
        """Visualize the marginalized learner posterior distribution over hypotheses"""
        self.hyp_learner_posterior = np.sum(self.learner_posterior, axis=1)

        # assert np.allclose(np.sum(graph_teacher.hyp_learner_posterior, axis=1), 1.0)

        actions = ['11', '12', '13', '22', '23', '33']
        hypotheses = ['CC1', 'CC2', 'CC3', 'CE1', 'CE2', 'CE3',
                      'CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH6']
        ind = np.arange(len(actions))

        plt.figure()

        for i in range(len(hypotheses)):
            plt.subplot(4, 3, i+1)
            plt.bar(ind, self.combine_actions(self.hyp_learner_posterior[i]))
            plt.xticks(ind, actions)
            plt.title(hypotheses[i])

        plt.show()


# run cooperative inference to teach graphs
graphs = create_graph_hyp_space(transmission_rate=0.9, background_rate=0.05)
graph_teacher = GraphTeacher(graphs)
graph_teacher.run_cooperative_inference()

graph_teacher.plot_learner_posterior_by_actions()
