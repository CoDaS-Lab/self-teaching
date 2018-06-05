import numpy as np
import matplotlib.pyplot as plt
from dag import DirectedGraph
from utils import create_graph_hyp_space

class GraphTeacher:
    def __init__(self, graphs, true_hyp=None):
        self.n_hyp = len(graphs)
        self.n_actions = 3
        self.n_observations = 8
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

        self.learner_posterior = self.learner_prior
        self.teacher_posterior = 1 / self.n_hyp * np.ones((self.n_hyp,
                                                           self.n_observations,
                                                           self.n_actions ** 2))

        # if no hypothesis is specified, select one at random
        if true_hyp is None:
            self.true_hyp = np.random.choice(self.n_hyp)
        else:
            self.true_hyp = true_hyp
        
    def likelihood(self):
        """Returns the likelihood of each action/outcome pair for each graph"""

        full_lik = np.empty((self.n_hyp, self.n_observations, self.n_actions ** 2))
        
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

        if self.teacher_posterior.shape == (12, 9):
            # reshape
            self.teacher_posterior = np.repeat(
                self.teacher_posterior[:, np.newaxis, :],
                self.n_observations, axis=1)

        post = self.teacher_posterior * self.likelihood()
        self.learner_posterior = np.nan_to_num(post / np.sum(post, axis=0))

    def update_teacher_posterior(self):
        """Calculates the posterior of selecting which actions to take"""
        joint_action_obs = 1 / (self.n_actions ** 2 * self.n_observations) * \
                           np.ones((self.n_hyp, self.n_observations, self.n_actions ** 2))

        joint_all = self.learner_posterior * joint_action_obs
        joint_actions = np.sum(joint_all, axis=1)
        self.teacher_posterior = (joint_actions.T / (np.sum(joint_actions, axis=1)).T).T

    def run_cooperative_inference(self, n_iters=1):
        for i in range(n_iters):
            self.update_learner_posterior()
            self.update_teacher_posterior()

graphs = create_graph_hyp_space()
graph_teacher = GraphTeacher(graphs)
graph_teacher.run_cooperative_inference()
bar = graph_teacher.teacher_posterior[0]
common_cause = [bar[0], bar[1] + bar[3], bar[2] + bar[6], bar[4], bar[5] + bar[7], bar[8]]

bar = graph_teacher.teacher_posterior[3]
common_effect = [bar[0], bar[1] + bar[3], bar[2] + bar[6], bar[4], bar[5] + bar[7], bar[8]]

bar = graph_teacher.teacher_posterior[6]
causal_chain = [bar[0], bar[1] + bar[3], bar[2] + bar[6], bar[4], bar[5] + bar[7], bar[8]]

actions = ['11', '12', '13', '22', '23', '33']
ind = np.arange(len(actions))

plt.figure()
plt.subplot(131)
plt.bar(ind, common_effect)
plt.xticks(ind, actions)
plt.title("Common effect")

plt.subplot(132)
plt.bar(ind, causal_chain)
plt.xticks(ind, actions)
plt.title("Causal chain")

plt.subplot(133)
plt.bar(ind, common_cause)
plt.xticks(ind, actions)
plt.title("Common cause")
plt.show()
