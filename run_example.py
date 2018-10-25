import numpy as np
import matplotlib.pyplot as plt
from models import utils
from models.concept_active_learner import ConceptActiveLearner
from models.concept_self_teacher import ConceptSelfTeacher
from models.graph_active_learner import GraphActiveLearner


def toy_first_feature_simulation():
    # function to check numbers for made up example
    # set up posterior
    toy_posterior = np.array([[[1/4, 0], [0, 1/2], [1/6, 1/3]],
                              [[1/4, 0], [0, 1/2], [1/6, 1/3]],
                              [[1/4, 0], [1/2, 0], [1/6, 1/3]],
                              [[1/4, 0], [1/2, 0], [1/2, 0]]])

    toy_likelihood = np.array([[[1, 0], [0, 1], [1/3, 2/3]],
                               [[1, 0], [0, 1], [1/3, 2/3]],
                               [[1, 0], [1, 0], [1/3, 2/3]],
                               [[1, 0], [1, 0], [1, 0]]])

    toy_prior = 1/4 * np.ones_like(toy_likelihood)

    # calculate self-teaching posterior
    # p_l(h)
    toy_self_teaching_prior = 1/2 * np.ones_like(toy_posterior)

    # p_t(x, y)
    toy_joint_feature_label_prior = 1/6 * np.ones_like(toy_posterior)

    # p(h|x, y) * p(x, y) / Z
    toy_self_teaching_posterior = toy_posterior * toy_joint_feature_label_prior
    toy_self_teaching_posterior = (toy_self_teaching_posterior.T /
                                   np.sum(toy_self_teaching_posterior, axis=(1, 2)).T).T
    toy_self_teaching_posterior = np.nan_to_num(toy_self_teaching_posterior)

    toy_self_teaching_posterior = np.sum(
        toy_self_teaching_posterior * toy_self_teaching_prior, axis=(0, 2))

    # normalize
    toy_self_teaching_posterior = toy_self_teaching_posterior / \
        np.sum(toy_self_teaching_posterior)

    print(toy_self_teaching_posterior)

    # calculate expected information gain
    # H(h)
    prior_entropy = -np.sum(toy_prior * np.log2(toy_prior), axis=0)

    # H(h|x, y)
    posterior_entropy = -np.sum(toy_posterior * np.nan_to_num(
        np.log2(toy_posterior)), axis=0)

    # p(y|x)
    toy_obs_lik = np.sum(toy_prior * toy_likelihood, axis=0)

    # EIG(x) = H(h) - \sum_y p(y|x) H(h|x, y)
    toy_eig = prior_entropy.T - np.sum(toy_obs_lik * posterior_entropy, axis=1)


def plot_first_feature_heatmap():
    import seaborn as sns
    hyp_space_type = "boundary"
    n_features = 3
    sampling = "max"

    # get predictions from self-teaching model
    st = ConceptSelfTeacher(n_features, hyp_space_type, sampling)
    st.update_learner_posterior()

    learner_posterior = np.array([[0, 1/3, 1/3, 1/3],
                                  [1, 0, 0, 0],
                                  [0, 0, 1/2, 1/2],
                                  [1/2, 1/2, 0, 0],
                                  [0, 0, 0, 1],
                                  [1/3, 1/3, 1/3, 0]])

    plt.figure()
    sns.heatmap(learner_posterior, cmap='Greys', linewidth=0.5, vmin=0, vmax=1)
    plt.xticks([1/2, 3/2, 5/2, 7/2], ['h1', 'h2', 'h3', 'h4'])
    plt.yticks([1/2, 3/2, 5/2, 7/2, 9/2, 11/2],
               ['x=1, y=0', 'x=1, y=1',
                'x=2, y=0', 'x=2, y=1',
                'x=3, y=0', 'x=3, y=1'],
               rotation=0)
    plt.tick_params(
        axis='both',
        which='both',
        bottom=False,
        left=False)
    plt.savefig('figures/example/matching_learner_posterior_heatmap.pdf')
    plt.close()

    teaching_posterior = np.array([
        [0, 2/7, 2/7, 2/11],
        [6/11, 0, 0, 0],
        [0, 0, 3/7, 3/11],
        [3/11, 3/7, 0, 0],
        [0, 0, 0, 6/11],
        [2/11, 2/7, 2/7, 0]
    ])

    plt.figure()
    sns.heatmap(teaching_posterior, cmap='Greys',
                linewidth=0.5, vmin=0, vmax=1)
    plt.xticks([1/2, 3/2, 5/2, 7/2], ['h1', 'h2', 'h3', 'h4'])
    plt.yticks([1/2, 3/2, 5/2, 7/2, 9/2, 11/2],
               ['x=1, y=0', 'x=1, y=1',
                'x=2, y=0', 'x=2, y=1',
                'x=3, y=0', 'x=3, y=1'],
               rotation=0)
    plt.tick_params(
        axis='both',
        which='both',
        bottom=False,
        left=False)
    plt.savefig('figures/example/matching_teaching_posterior_heatmap.pdf')
    plt.close()

    learner_posterior_colsum = np.array([11/6, 7/6, 7/6, 11/6])
    plt.figure()
    sns.barplot(np.arange(4), learner_posterior_colsum, palette='Greys')
    plt.xticks(np.arange(4), ['h1', 'h2', 'h3', 'h4'])
    plt.savefig('figures/example/matching_learner_posterior_colsum.pdf')

    learner_posterior_rowsum = np.array([1, 1, 1, 1, 1, 1])
    plt.figure()
    sns.barplot(np.arange(6), learner_posterior_rowsum, palette='Greys')
    plt.xticks(np.arange(6), ['x=1, y=0', 'x=1, y=1', 'x=2, y=0',
                              'x=2, y=1', 'x=3, y=0', 'x=3, y=1'])
    plt.savefig('figures/example/matching_learner_posterior_rowsum.pdf')

    # calculate 1/Z
    Z = np.sum(learner_posterior, axis=0) * 1/6
    Z_inv = 1 / Z
    plt.figure()
    sns.barplot(np.arange(4), Z_inv, palette='Greys')
    plt.xticks(np.arange(4), ['h1', 'h2', 'h3', 'h4'])
    plt.savefig('figures/example/matching_Z_inv.pdf')

    self_teaching_posterior = np.array([25/77, 27/77, 25/77])
    plt.figure()
    sns.barplot(np.arange(3), self_teaching_posterior, palette='Greys')
    plt.xticks(np.arange(3), ['x1', 'x2', 'x3'])
    plt.savefig('figures/example/matching_self_teaching_posterior.pdf')

    obs_lik = np.sum(
        st.likelihood() * st.learner_prior, axis=0)
    plt.figure()
    sns.barplot(np.arange(6), obs_lik.flatten(), palette='Greys')
    plt.xticks(np.arange(6), ['x=1, y=0', 'x=1, y=1', 'x=2, y=0',
                              'x=2, y=1', 'x=3, y=0', 'x=3, y=1'])
    plt.savefig('figures/example/matching_observation_likelihood.pdf')

    prior_entropy = np.array([2, 2, 2])
    posterior_entropy = -np.sum(st.learner_posterior *
                                np.nan_to_num(np.log2(st.learner_posterior)), axis=0)
    plt.figure()
    sns.barplot(np.arange(6), posterior_entropy.flatten(), palette='Greys')
    plt.xticks(np.arange(6), ['x=1, y=0', 'x=1, y=1', 'x=2, y=0',
                              'x=2, y=1', 'x=3, y=0', 'x=3, y=1'])
    plt.savefig('figures/example/matching_posterior_entropy.pdf')

    expected_information_gain = prior_entropy.T - \
        np.sum(obs_lik * posterior_entropy, axis=1)
    expected_information_gain = expected_information_gain / \
        np.sum(expected_information_gain)
    plt.figure()
    sns.barplot(np.arange(3), expected_information_gain, palette='Greys')
    plt.xticks(np.arange(3), ['x1', 'x2', 'x3'])
    plt.savefig('figures/example/matching_expected_information_gain.pdf')

    prior_entropy = np.array([2, 2, 2, 2, 2, 2])
    information_gain = prior_entropy - posterior_entropy.flatten()
    plt.figure()
    sns.barplot(np.arange(6), information_gain, palette='Greys')
    plt.xticks(np.arange(6), ['x=1, y=0', 'x=1, y=1', 'x=2, y=0',
                              'x=2, y=1', 'x=3, y=0', 'x=3, y=1'])
    plt.savefig('figures/example/matching_information_gain.pdf')
