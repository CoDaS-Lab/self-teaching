import numpy as np
import matplotlib.pyplot as plt

from models.active_learner import ActiveLearner
from models.random_learner import RandomLearner
from models.teacher import Teacher
from models.self_teacher import SelfTeacher
from models.teacher_rollout import TeacherRollout
from models.bayesian_learner import BayesianLearner
from models.hypothesis_dependent_learner import HypothesisDependentLearner

if __name__ == "__main__":
    n_features = 8
    n_iters = 1000

    # save number of observations for each iteration
    active_obs = np.zeros(n_iters)
    random_obs = np.zeros(n_iters)
    teacher_obs = np.zeros(n_iters)
    self_teacher_obs = np.zeros(n_iters)
    bayesian_learner_obs = np.zeros(n_iters)
    # hypothesis_dependent_obs = np.zeros(n_iters)

    # save posterior probability for each iteration
    active_post = np.zeros((n_iters, n_features + 1))
    random_post = np.zeros((n_iters, n_features + 1))
    teacher_post = np.zeros((n_iters, n_features + 1))
    self_teacher_post = np.zeros((n_iters, n_features + 1))
    bayesian_learner_post = np.zeros((n_iters, n_features + 1))
    # hypothesis_dependent_post = np.zeros((n_iters, n_features + 1))

    hyp_space_type = "boundary"

    for i in range(n_iters):
        # create active, teacher and random learners
        active_learner = ActiveLearner(n_features, hyp_space_type)
        random_learner = RandomLearner(n_features, hyp_space_type)
        teacher = Teacher(n_features, hyp_space_type)
        self_teacher = SelfTeacher(n_features, hyp_space_type)
        bayesian_learner = BayesianLearner(n_features, hyp_space_type)
        # hypothesis_dependent_learner = HypothesisDependentLearner(
        #     n_features, hyp_space_type)

        # run simulations across all models
        active_obs[i], active_post[i, :] = active_learner.run()
        random_obs[i], random_post[i, :] = random_learner.run()
        teacher_obs[i], teacher_post[i, :] = teacher.run()
        self_teacher_obs[i], self_teacher_post[i, :] = self_teacher.run()
        bayesian_learner_obs[i], bayesian_learner_post[i,
                                                       :] = bayesian_learner.run()
        # hypothesis_dependent_obs[i], hypothesis_dependent_post[i,
        #                                                        :] =
        # hypothesis_dependent_learner.run()

    # calculate mean posterior probability of true hypothesis
    active_post_mean = np.mean(active_post, axis=0)
    random_post_mean = np.mean(random_post, axis=0)
    teacher_post_mean = np.mean(teacher_post, axis=0)
    self_teacher_post_mean = np.mean(self_teacher_post, axis=0)
    bayesian_learner_post_mean = np.mean(bayesian_learner_post, axis=0)
    # hypothesis_dependent_post_mean = np.mean(hypothesis_dependent_post, axis=0)

    # calculate cumulative probability of finding true hypothesis
    # active_cumulative_obs = [
    #     np.sum(active_obs <= i) / n_iters for i in range(n_features + 1)]
    # random_cumulative_obs = [
    #     np.sum(random_obs <= i) / n_iters for i in range(n_features + 1)]
    # teacher_cumulative_obs = [
    #     np.sum(teacher_obs <= i) / n_iters for i in range(n_features + 1)]
    # self_teacher_cumulative_obs = [
    #     np.sum(self_teacher_obs <= i) / n_iters for i in range(n_features + 1)]
    # bayesian_learner_cumulative_obs = [
    #     np.sum(bayesian_learner_obs <= i) / n_iters for i in range(n_features + 1)]
    # hypothesis_dependent_cumulative_obs = [
    #     np.sum(hypothesis_dependent_obs <= i) / n_iters for i in range(n_features + 1)]

    features = np.arange(n_features + 1)

    # # plot each run individually
    # for i in range(n_iters):
    #     plt.plot(features, active_post[i, :],
    #              '-o', alpha=0.01, color='#1f77b4')
    #     plt.plot(features, random_post[i, :],
    #              '-o', alpha=0.01, color='#ff7f0e')
    #     plt.plot(features, teacher_post[i, :],
    #              '-o', alpha=0.01, color='#2ca02c')
    #     plt.plot(features, self_teacher_post[i, :],
    #              '-o', alpha=0.01, color='#d62728')
    #     plt.plot(
    #         features, bayesian_learner_post[i, :], '-o', alpha=0.01, color='#9467bd')

    # plt.show()

    plt.plot(features, active_post_mean, '-o', label='Active Learner')
    plt.plot(features, random_post_mean, '-o', label='Random Learner')
    plt.plot(features, teacher_post_mean, '-o', label='Teaching')
    plt.plot(features, self_teacher_post_mean,
             '-o', label='Self-teaching')
    plt.plot(features, bayesian_learner_post_mean,
             '-o', label='Weak sampling')
    # plt.plot(features, hypothesis_dependent_post_mean,
    #          '-o', label='Hypothesis Dependent Learner')
    plt.xlabel("Number of features observed")
    plt.ylabel("Posterior probability of the true hypothesis")
    plt.legend()
    plt.show()

    # active_learner_counts = np.bincount(active_obs.astype(int)) / n_iters
    # random_learner_counts = np.bincount(random_obs.astype(int)) / n_iters
    # teacher_counts = np.bincount(teacher_obs.astype(int)) / n_iters

    # plt.bar(features, active_learner_counts, 0.2)
    # plt.bar(features + 0.2, random_learner_counts, 0.2)
    # plt.bar(features + 0.4, teacher_counts, 0.2)
    # plt.show()
    # for i in range(n_iters):
    #     plt.plot(features, random_post[i, :], '-ro', alpha=0.1)

    # plt.show()
