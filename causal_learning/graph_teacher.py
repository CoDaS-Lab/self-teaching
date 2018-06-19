import numpy as np
import matplotlib.pyplot as plt
# from utils import create_graph_hyp_space


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
        self.hyp = graphs

        # prior over graphs
        self.learner_prior = 1 / self.n_hyp * np.ones((self.n_hyp,
                                                       self.n_observations))
        # prior over teaching actions
        self.teacher_prior = (1 / self.n_actions) * \
            np.ones((self.n_hyp,
                     self.n_observations))

        # initialize posteriors to be over the priors
        self.learner_posterior = self.learner_prior
        self.teacher_posterior = self.teacher_prior

    def likelihood(self):
        """Returns the likelihood of each action/outcome pair for each graph"""

        full_lik = np.empty((self.n_hyp,
                             self.n_observations))

        for i, h in enumerate(self.hyp):
            lik = h.likelihood

            full_lik[:, i] = lik

        self.lik = full_lik

    def marginal_likelihood(self, prior):
        self.M = np.zeros((self.n_hyp, self.n_observations))
        interventions = [[0, 1, 2, 3], [4, 5, 8, 9], [6, 7, 10, 11]]

        for intervention in interventions:
            denom = np.sum(self.lik[intervention] *
                           prior[intervention], axis=1)

            # TODO: deal with zeros

            tmp = ((self.lik[intervention] *
                    prior[intervention]).T / denom).T

            tmp = np.sum(tmp, axis=0)
            tmp = tmp / np.sum(tmp)
            self.M[intervention, :] = np.tile(tmp, (len(intervention), 1))

        uniqueInt = [3, 9, 11]
        new = self.M[uniqueInt] / np.sum(self.M[uniqueInt], axis=0)
        self.M = new[[0, 0, 0, 0, 1, 1, 2, 2, 1, 1, 2, 2]]

    def sequential_likelihood(self):
        # calculate updated prior
        self.sequential_prior = np.zeros((self.n_interventions,
                                          self.n_hyp,
                                          self.n_observations))
        self.sequential_likelihood = np.zeros((self.n_interventions,
                                               self.n_hyp,
                                               self.n_observations))

        for i in range(self.n_interventions):
            prior_two = np.sum(
                self.learner_posterior[self.interventions == i], axis=0)
            prior_two = prior_two / np.sum(prior_two)
            self.sequential_prior[i] = np.tile(
                prior_two, (self.n_observations, 1))

            self.marginal_likelihood(self.sequential_prior[i])
            self.sequential_likelihood[i] = graph_teacher.compute_likelihood(
                self.sequential_prior[i])

    def compute_likelihood(self, prior):
        """Run cooperative inference"""

        crit = 0.00001
        M1 = np.zeros_like(self.M)
        M = self.M.copy()

        while np.sum(np.absolute(M - M1)) > crit:
            M1 = M.copy()
            M = self.f(M, prior)

            uniqueInt = [3, 9, 11]
            new = M[uniqueInt] / np.sum(M[uniqueInt], axis=0)
            M = new[[0, 0, 0, 0, 1, 1, 2, 2, 1, 1, 2, 2]]

        return M

    def f(self, M, prior):
        interventions = [[0, 1, 2, 3], [4, 5, 8, 9], [6, 7, 10, 11]]

        for intervention in interventions:
            denom = np.sum(M[intervention] * self.lik[intervention] *
                           prior[intervention], axis=1)

            tmp = ((M[intervention] *
                    prior[intervention]).T / denom).T

            tmp = np.sum(tmp, axis=0)
            tmp = tmp / np.sum(tmp)
            M[intervention, :] = np.tile(tmp, (len(intervention), 1))

        M = (M.T / np.sum(M, axis=1)).T
        return M

    def update_learner_posterior(self):
        posterior = self.M * self.lik * self.learner_prior
        self.learner_posterior = (posterior.T / np.sum(posterior, axis=1)).T

    def update_sequential_learner_posterior(self):
        pass

        # def update_learner_posterior(self):
        #     """Calculates the posterior over all possible action/outcome pairs
        #     for each graph"""

        #     # p(g|a, o) = p(o, a|g) * p(g)
        #     post = self.likelihood() * self.teacher_posterior * self.learner_posterior
        #     self.learner_posterior = np.nan_to_num(post / np.sum(post, axis=0))

        #     # check that learner's posterior is a valid distribution
        #     # TODO: figure out how to check that a numpy array has both 0s and 1s

    def update_teacher_posterior(self):
        """Calculates the posterior of selecting which actions to take"""
        # p(a, o) = 1 / (|a|^2 * |o|)
        joint_action_obs = 1 / (self.n_actions ** 2 * self.n_observations) * \
            np.ones((self.n_hyp,
                     self.n_observations,
                     self.n_actions ** 2))

        # p(g, a, o) = p(g|a, o) * p(a, o)
        joint_all = self.learner_posterior * joint_action_obs

        # p(g, a) = \sum_o p(g, a, o)
        joint_actions = np.sum(joint_all, axis=1)

        # p(a|g) = p(g, a) / p(g) = p(g, a) / \sum_a p(g, a)
        self.teacher_posterior = (joint_actions.T /
                                  (np.sum(joint_actions, axis=1)).T).T

        # expand teacher's posterior to add observation dimension back in
        self.teacher_posterior = np.repeat(
            self.teacher_posterior[:, np.newaxis, :],
            self.n_observations, axis=1)

        # check that teacher's posterior is a valid distribution over actions
        assert np.allclose(np.sum(self.teacher_posterior, axis=2), 1.0)

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

    def plot_likelihood(self):
        ex_num = [0, 4, 6]
        cause_num = [0, 3, 6]
        teach = np.zeros((cause_num, ex_num, ex_num))

        for i, cause in enumerate(cause_num):
            for j, ex_one in enumerate(ex_num):
                for k, ex_two in enumerate(ex_num):
                    teach[i, j, k] = self.lik[ex_one, cause] * \
                        self.sequential_likelihood[ex_one, ex_two, cause]

    def plot_learner_posterior_by_hypotheses(self):
        """Visualize the marginalized learner posterior distribution over hypotheses"""
        self.hyp_learner_posterior = np.sum(self.learner_posterior, axis=1)

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

    def plot_teacher_posterior(self):
        """Visualize the teaching posterior over the three possible kinds of causal graphs"""

        # marginalize over observations and normalize
        marg_teaching_posterior = np.sum(self.teacher_posterior, axis=1)
        marg_teaching_posterior = (marg_teaching_posterior.T /
                                   np.sum(marg_teaching_posterior, axis=1).T).T

        # get three canonical causal graphs
        common_cause = self.combine_actions(marg_teaching_posterior[0])
        common_effect = self.combine_actions(marg_teaching_posterior[3])
        causal_chain = self.combine_actions(marg_teaching_posterior[6])

        # make figures
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


# run cooperative inference to teach graphs
graphs = create_graph_hyp_space(t=0.8, b=0.01)
graph_teacher = GraphTeacher(graphs)
graph_teacher.likelihood()
graph_teacher.marginal_likelihood(graph_teacher.learner_prior)
lik = graph_teacher.M
graph_teacher.update_learner_posterior()

graph_teacher.sequential_likelihood()

ex_num = [0, 4, 6]
cause_num = [0, 3, 6]
teach = np.zeros((len(cause_num), len(ex_num), len(ex_num)))

for i, cause in enumerate(cause_num):
    for j, ex_one in enumerate(ex_num):
        for k, ex_two in enumerate(ex_num):
            teach[i, j, k] = lik[ex_one, cause] * \
                graph_teacher.sequential_likelihood[j, ex_two, cause]

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

# interventions = np.array([0, 0, 0, 0, 1, 1, 2, 2, 1, 1, 2, 2])
# prior_two = np.sum(graph_teacher.learner_posterior[interventions == 0], axis=0)
# prior_two = prior_two / np.sum(prior_two)
# prior_two = np.tile(prior_two, (12, 1))

# graph_teacher.marginal_likelihood(prior_two)
# graph_teacher.compute_likelihood(prior_two)

# M1 = graph_teacher.M.copy()
# print(graph_teacher.M)

# interventions = [[0, 1, 2, 3], [4, 5, 8, 9], [6, 7, 10, 11]]

# for intervention in interventions:
#     denom = np.sum(graph_teacher.M[intervention] * graph_teacher.lik[intervention] *
#                    prior_two[intervention], axis=1)

#     tmp = ((graph_teacher.M[intervention] *
#             prior_two[intervention]).T / denom).T

#     tmp = np.sum(tmp, axis=0)
#     tmp = tmp / np.sum(tmp)
#     graph_teacher.M[intervention, :] = np.tile(tmp, (len(intervention), 1))

# graph_teacher.M = (graph_teacher.M.T / np.sum(graph_teacher.M, axis=1)).T

# uniqueInt = [3, 9, 11]
# new = graph_teacher.M[uniqueInt] / np.sum(graph_teacher.M[uniqueInt], axis=0)
# graph_teacher.M = new[[0, 0, 0, 0, 1, 1, 2, 2, 1, 1, 2, 2]]
# graph_teacher.compute_likelihood(prior_two)
# np.all(np.isclose(lik, pDgHI))
# print(graph_teacher.likelihood())
# graph_teacher.run_cooperative_inference(n_iters=3)
# graph_teacher.plot_teacher_posterior()
