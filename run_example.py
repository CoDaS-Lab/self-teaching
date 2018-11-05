import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from models.concept_self_teacher import ConceptSelfTeacher


def plot_mismatching_example_figures():
    # function to check numbers for made up example
    hyp_space_type = "boundary"
    n_features = 3
    sampling = "max"

    # get predictions from self-teaching model
    st = ConceptSelfTeacher(n_features, hyp_space_type, sampling)
    st.learner_posterior = np.array([[[1/4, 0], [0, 1/2], [1/6, 1/3]],
                                     [[1/4, 0], [0, 1/2], [1/6, 1/3]],
                                     [[1/4, 0], [1/2, 0], [1/6, 1/3]],
                                     [[1/4, 0], [1/2, 0], [1/2, 0]]])

    learner_posterior_flat = st.learner_posterior.reshape(4, 6).T

    plt.figure()
    sns.heatmap(learner_posterior_flat, cmap='Greys',
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
    plt.savefig('figures/example/mismatching_learner_posterior_heatmap.pdf')
    plt.close()

    teaching_posterior = st.learner_posterior * 1/4
    denom = np.sum(teaching_posterior, axis=(1, 2))
    teaching_posterior = (teaching_posterior.T / denom.T).T
    teaching_posterior_flat = teaching_posterior.reshape(4, 6).T

    plt.figure()
    sns.heatmap(teaching_posterior_flat, cmap='Greys',
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
    plt.savefig('figures/example/mismatching_teaching_posterior_heatmap.pdf')
    plt.close()

    learner_posterior_colsum = np.sum(learner_posterior_flat, axis=0)
    plt.figure()
    sns.barplot(np.arange(4), learner_posterior_colsum, palette='Greys')
    plt.xticks(np.arange(4), ['h1', 'h2', 'h3', 'h4'])
    plt.savefig('figures/example/mismatching_learner_posterior_colsum.pdf')

    learner_posterior_rowsum = np.sum(learner_posterior_flat, axis=1)
    plt.figure()
    sns.barplot(np.arange(6), learner_posterior_rowsum, palette='Greys')
    plt.xticks(np.arange(6), ['x=1, y=0', 'x=1, y=1', 'x=2, y=0',
                              'x=2, y=1', 'x=3, y=0', 'x=3, y=1'])
    plt.savefig('figures/example/mismatching_learner_posterior_rowsum.pdf')

    teaching_posterior_rowsum = np.sum(teaching_posterior_flat, axis=1)
    plt.figure()
    sns.barplot(np.arange(6), teaching_posterior_rowsum, palette='Greys')
    plt.xticks(np.arange(6), ['x=1, y=0', 'x=1, y=1', 'x=2, y=0',
                              'x=2, y=1', 'x=3, y=0', 'x=3, y=1'])
    plt.savefig('figures/example/mismatching_teaching_posterior_rowsum.pdf')

    # calculate 1/Z
    Z = np.sum(learner_posterior_flat, axis=0) * 1/6
    Z_inv = 1 / Z
    plt.figure()
    sns.barplot(np.arange(4), Z_inv, palette='Greys')
    plt.xticks(np.arange(4), ['h1', 'h2', 'h3', 'h4'])
    plt.savefig('figures/example/mismatching_Z_inv.pdf')

    self_teaching_posterior = np.sum(teaching_posterior * 1/4, axis=(0, 2))
    plt.figure()
    sns.barplot(np.arange(3), self_teaching_posterior, palette='Greys')
    plt.xticks(np.arange(3), ['x1', 'x2', 'x3'])
    plt.savefig('figures/example/mismatching_self_teaching_posterior.pdf')

    # hard code new likelihood
    likelihood_flat = np.array([[1, 1, 1, 1],
                                [0, 0, 0, 0],
                                [0, 0, 1, 1],
                                [1, 1, 0, 0],
                                [1/3, 1/3, 1/3, 1],
                                [2/3, 2/3, 2/3, 0]])

    obs_lik = np.sum(likelihood_flat * 1/4, axis=1)
    plt.figure()
    sns.barplot(np.arange(6), obs_lik, palette='Greys')
    plt.xticks(np.arange(6), ['x=1, y=0', 'x=1, y=1', 'x=2, y=0',
                              'x=2, y=1', 'x=3, y=0', 'x=3, y=1'])
    plt.savefig('figures/example/mismatching_observation_likelihood.pdf')

    prior_entropy = np.array([2, 2, 2])
    posterior_entropy = -np.sum(learner_posterior_flat *
                                np.nan_to_num(np.log2(learner_posterior_flat)), axis=1)
    plt.figure()
    sns.barplot(np.arange(6), posterior_entropy, palette='Greys')
    plt.xticks(np.arange(6), ['x=1, y=0', 'x=1, y=1', 'x=2, y=0',
                              'x=2, y=1', 'x=3, y=0', 'x=3, y=1'])
    plt.savefig('figures/example/mismatching_posterior_entropy.pdf')

    prior_entropy = np.array([2, 2, 2])
    weighted_posterior_entropy = np.array([
        obs_lik[0] * posterior_entropy[0] + obs_lik[1] * posterior_entropy[1],
        obs_lik[2] * posterior_entropy[2] + obs_lik[3] * posterior_entropy[3],
        obs_lik[4] * posterior_entropy[4] + obs_lik[5] * posterior_entropy[5],
    ])
    expected_information_gain = prior_entropy - weighted_posterior_entropy
    expected_information_gain = expected_information_gain / \
        np.sum(expected_information_gain)
    plt.figure()
    sns.barplot(np.arange(3), expected_information_gain, palette='Greys')
    plt.xticks(np.arange(3), ['x1', 'x2', 'x3'])
    plt.savefig('figures/example/mismatching_expected_information_gain.pdf')

    prior_entropy = np.array([2, 2, 2, 2, 2, 2])
    information_gain = prior_entropy - posterior_entropy
    plt.figure()
    sns.barplot(np.arange(6), information_gain, palette='Greys')
    plt.xticks(np.arange(6), ['x=1, y=0', 'x=1, y=1', 'x=2, y=0',
                              'x=2, y=1', 'x=3, y=0', 'x=3, y=1'])
    plt.savefig('figures/example/mismatching_information_gain.pdf')


def plot_matching_example_figures():
    hyp_space_type = "boundary"
    n_features = 3
    sampling = "max"

    # get predictions from self-teaching model
    st = ConceptSelfTeacher(n_features, hyp_space_type, sampling)
    st.update_learner_posterior()

    learner_posterior_flat = st.learner_posterior.reshape(4, 6).T

    plt.figure()
    sns.heatmap(learner_posterior_flat, cmap='Greys',
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
    plt.savefig('figures/example/matching_learner_posterior_heatmap.pdf')
    plt.close()

    teaching_posterior = st.learner_posterior * 1/4
    denom = np.sum(teaching_posterior, axis=(1, 2))
    teaching_posterior = (teaching_posterior.T / denom.T).T
    teaching_posterior_flat = teaching_posterior.reshape(4, 6).T

    plt.figure()
    sns.heatmap(teaching_posterior_flat, cmap='Greys',
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

    learner_posterior_colsum = np.sum(learner_posterior_flat, axis=0)
    plt.figure()
    sns.barplot(np.arange(4), learner_posterior_colsum, palette='Greys')
    plt.xticks(np.arange(4), ['h1', 'h2', 'h3', 'h4'])
    plt.savefig('figures/example/matching_learner_posterior_colsum.pdf')

    learner_posterior_rowsum = np.sum(learner_posterior_flat, axis=1)
    plt.figure()
    sns.barplot(np.arange(6), learner_posterior_rowsum, palette='Greys')
    plt.xticks(np.arange(6), ['x=1, y=0', 'x=1, y=1', 'x=2, y=0',
                              'x=2, y=1', 'x=3, y=0', 'x=3, y=1'])
    plt.savefig('figures/example/matching_learner_posterior_rowsum.pdf')

    teaching_posterior_rowsum = np.sum(teaching_posterior_flat, axis=1)
    plt.figure()
    sns.barplot(np.arange(6), teaching_posterior_rowsum, palette='Greys')
    plt.xticks(np.arange(6), ['x=1, y=0', 'x=1, y=1', 'x=2, y=0',
                              'x=2, y=1', 'x=3, y=0', 'x=3, y=1'])
    plt.savefig('figures/example/matching_teaching_posterior_rowsum.pdf')

    # calculate 1/Z
    Z = np.sum(learner_posterior_flat, axis=0) * 1/6
    Z_inv = 1 / Z
    plt.figure()
    sns.barplot(np.arange(4), Z_inv, palette='Greys')
    plt.xticks(np.arange(4), ['h1', 'h2', 'h3', 'h4'])
    plt.savefig('figures/example/matching_Z_inv.pdf')

    self_teaching_posterior = np.sum(teaching_posterior * 1/4, axis=(0, 2))
    plt.figure()
    sns.barplot(np.arange(3), self_teaching_posterior, palette='Greys')
    plt.xticks(np.arange(3), ['x1', 'x2', 'x3'])
    plt.savefig('figures/example/matching_self_teaching_posterior.pdf')

    likelihood_flat = st.likelihood().reshape(4, 6).T

    obs_lik = np.sum(likelihood_flat * 1/4, axis=1)
    plt.figure()
    sns.barplot(np.arange(6), obs_lik, palette='Greys')
    plt.xticks(np.arange(6), ['x=1, y=0', 'x=1, y=1', 'x=2, y=0',
                              'x=2, y=1', 'x=3, y=0', 'x=3, y=1'])
    plt.savefig('figures/example/matching_observation_likelihood.pdf')

    prior_entropy = np.array([2, 2, 2])
    posterior_entropy = -np.sum(learner_posterior_flat *
                                np.nan_to_num(np.log2(learner_posterior_flat)), axis=1)
    plt.figure()
    sns.barplot(np.arange(6), posterior_entropy, palette='Greys')
    plt.xticks(np.arange(6), ['x=1, y=0', 'x=1, y=1', 'x=2, y=0',
                              'x=2, y=1', 'x=3, y=0', 'x=3, y=1'])
    plt.savefig('figures/example/matching_posterior_entropy.pdf')

    prior_entropy = np.array([2, 2, 2])
    weighted_posterior_entropy = np.array([
        obs_lik[0] * posterior_entropy[0] + obs_lik[1] * posterior_entropy[1],
        obs_lik[2] * posterior_entropy[2] + obs_lik[3] * posterior_entropy[3],
        obs_lik[4] * posterior_entropy[4] + obs_lik[5] * posterior_entropy[5],
    ])
    expected_information_gain = prior_entropy - weighted_posterior_entropy
    expected_information_gain = expected_information_gain / \
        np.sum(expected_information_gain)
    plt.figure()
    sns.barplot(np.arange(3), expected_information_gain, palette='Greys')
    plt.xticks(np.arange(3), ['x1', 'x2', 'x3'])
    plt.savefig('figures/example/matching_expected_information_gain.pdf')

    prior_entropy = np.array([2, 2, 2, 2, 2, 2])
    information_gain = prior_entropy - posterior_entropy
    plt.figure()
    sns.barplot(np.arange(6), information_gain, palette='Greys')
    plt.xticks(np.arange(6), ['x=1, y=0', 'x=1, y=1', 'x=2, y=0',
                              'x=2, y=1', 'x=3, y=0', 'x=3, y=1'])
    plt.savefig('figures/example/matching_information_gain.pdf')
