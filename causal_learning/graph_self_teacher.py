import numpy as np
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
            lik[i] = h.lik

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
        print(self.learner_posterior)
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

        print(self_teaching_posterior_original)

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

    # example one
    common_cause_1 = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]])
    common_cause_1_lik = np.array(
        [((1-t)*(1-b))**2, (1-t)*(1-b)*(t + (1-t)*b), (t + (1-t)*b)*(1-t)*(1-b), (t + (1-t)*b)**2,
         (1-b)**2, (1-b)*b, (1-b)**2, (1-b)*b,
         b*(1-t)*(1-b), b*(t + (1-t)*b), b*(1-t)*(1-b), b*(t + (1-t)*b)])

    common_cause_2 = np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0]])
    common_cause_2_lik = utils.permute_likelihood(
        common_cause_1_lik, (2, 1, 3))
    common_cause_2_lik[7], common_cause_2_lik[10] = \
        common_cause_2_lik[10], common_cause_2_lik[7]

    graphs_one = [common_cause_2, common_cause_1]
    likelihoods_one = [common_cause_2_lik, common_cause_1_lik]
    common_cause_graphs = [dag.DirectedGraph(graph, likelihood, t, b)
                           for (graph, likelihood) in zip(graphs_one, likelihoods_one)]

    np.set_printoptions(suppress=True)

    graphs = utils.create_graph_hyp_space(t=0.8, b=0.01)

    gt = GraphTeacher(graphs)
    gt.likelihood()
    teacher_posterior = gt.update_teacher_posterior(
        gt.learner_prior)

    gst = GraphSelfTeacher(common_cause_graphs)
    gst.update_learner_posterior()
    lik = gst.likelihood()
    self_teacher_posterior = gst.update_self_teaching_posterior()

    # post = self_teacher_posterior[:, gst.unique_interventions]

    causal_chain_1 = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    causal_chain_1_lik = np.array(
        [(1-t)*(1-b)*(1-b), (1-t)*(1-b)*b, (t + (1-t)*b)*(1-t)*(1-b), (t + (1-t)*b)**2,
         (1-b)*(1-t)*(1-b), (1-b)*(t + (1-t)*b), (1-b)**2, (1-b)*b,
            b*(1 - t)*(1-b), b*(t + (1-t)*b), b*(1-t)*(1-b), b*(t + (1-t)*b)]
    )

    causal_chain_2 = np.array([[0, 0, 1], [0, 0, 0], [0, 1, 0]])
    causal_chain_2_lik = utils.permute_likelihood(
        causal_chain_1_lik, (1, 3, 2))
    causal_chain_2_lik[1], causal_chain_2_lik[2] = \
        causal_chain_2_lik[2], causal_chain_2_lik[1]

    graphs_two = [causal_chain_2, causal_chain_1]
    likelihoods_two = [causal_chain_2_lik, causal_chain_1_lik]
    causal_chain_graphs = [dag.DirectedGraph(graph, likelihood, t, b)
                           for (graph, likelihood) in zip(graphs_two, likelihoods_two)]

    gst = GraphSelfTeacher(causal_chain_graphs)
    gst.update_learner_posterior()
    lik = gst.likelihood()
    self_teacher_posterior = gst.update_self_teaching_posterior()

    common_effect_1 = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]])
    common_effect_1_lik = np.array(
        [(1-b)*(1-t)*(1-b), (1-b)*(t + (1-t)*b), b*(1-t)*(1-t)*(1-b),
         b*(t*(1-t) + (1-t)*t + t*t + b*((1-t)**2)),
         (1-b)*(1-t)*(1-b), (1-b)*(t + (1-t)*b), (1-b)*(1-b), (1-b)*b,
         b*(1-t)*(1-t)*(1-b), b*(t*(1-t) + (1-t)*t + t*t + b*((1-t)**2)), (1-b)*b, b*b])

    common_effect_2 = np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]])
    common_effect_2_lik = utils.permute_likelihood(
        common_effect_1_lik, (3, 1, 2))
    common_effect_2_lik[1], common_effect_2_lik[2] = \
        common_effect_2_lik[2], common_effect_2_lik[1]

    single_link = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
    single_link_lik = np.array([(1-t)*(1-b)*(1-b), (1-t)*(1-b)*b,
                                (t + (1-t)*b)*(1-b), (t + (1-t)*b)*b,
                                (1-b)*(1-b), (1-b)*b,
                                (1-b)*(1-b), (1-b)*b,
                                b*(1-b), b*b,
                                b*(1-t)*(1-b), b*(t + (1-t)*b)])

    graphs_three = [common_effect_2, single_link]
    likelihoods_three = [common_effect_2_lik, single_link_lik]
    ex_three_graphs = [dag.DirectedGraph(graph, likelihood, t, b)
                       for (graph, likelihood) in zip(graphs_three, likelihoods_three)]

    gst = GraphSelfTeacher(ex_three_graphs)
    gst.update_learner_posterior()
    lik = gst.likelihood()
    self_teacher_posterior = gst.update_self_teaching_posterior()
