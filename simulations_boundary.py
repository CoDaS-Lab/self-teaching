import numpy as np
import matplotlib.pyplot as plt

from models.active_learner import ActiveLearner
from models.random_learner import RandomLearner
from models.teacher import Teacher
from models.self_teacher import SelfTeacher
from models.teacher_rollout import TeacherRollout
from models.bayesian_learner import BayesianLearner

if __name__ == "__main__":
    n_features = 8
    n_iters = 1000

    # save number of observations for each iteration
    active_obs = np.zeros(n_iters)
    random_obs = np.zeros(n_iters)
    teacher_obs = np.zeros(n_iters)
    self_teacher_obs = np.zeros(n_iters)
    bayesian_learner_obs = np.zeros(n_iters)
    # rollout_none_obs = np.zeros(n_iters)
    # rollout_one_obs = np.zeros(n_iters)
    # rollout_two_obs = np.zeros(n_iters)
    # rollout_three_obs = np.zeros(n_iters)
    # rollout_four_obs = np.zeros(n_iters)
    # rollout_full_obs = np.zeros(n_iters)

    # save posterior probability for each iteration
    active_post = np.zeros((n_iters, n_features + 1))
    random_post = np.zeros((n_iters, n_features + 1))
    teacher_post = np.zeros((n_iters, n_features + 1))
    self_teacher_post = np.zeros((n_iters, n_features + 1))
    bayesian_learner_post = np.zeros((n_iters, n_features + 1))
    # rollout_none_post = np.zeros((n_iters, n_features + 1))
    # rollout_one_post = np.zeros((n_iters, n_features + 1))
    # rollout_two_post = np.zeros((n_iters, n_features + 1))
    # rollout_three_post = np.zeros((n_iters, n_features + 1))
    # rollout_four_post = np.zeros((n_iters, n_features + 1))
    # rollout_full_post = np.zeros((n_iters, n_features + 1))

    hyp_space_type = "boundary"

    for i in range(n_iters):
        # create active, teacher and random learners
        active_learner = ActiveLearner(n_features, hyp_space_type)
        random_learner = RandomLearner(n_features, hyp_space_type)
        teacher = Teacher(n_features, hyp_space_type)
        self_teacher = SelfTeacher(n_features, hyp_space_type)
        bayesian_learner = BayesianLearner(n_features, hyp_space_type)
        # rollout_none = TeacherRollout(n_features, 0)
        # rollout_one = TeacherRollout(n_features, 1)
        # rollout_two = TeacherRollout(n_features, 2)
        # rollout_three = TeacherRollout(n_features, 3)
        # rollout_four = TeacherRollout(n_features, 4)
        # rollout_full = TeacherRollout(n_features, n_features)

        # run simulations across all models
        active_obs[i], active_post[i, :] = active_learner.run()
        random_obs[i], random_post[i, :] = random_learner.run()
        teacher_obs[i], teacher_post[i, :] = teacher.run()
        self_teacher_obs[i], self_teacher_post[i, :] = self_teacher.run()
        bayesian_learner_obs[i], bayesian_learner_post[i,
                                                       :] = bayesian_learner.run()
        # rollout_none_obs[i], rollout_none_post[i, :] = rollout_none.run()
        # rollout_one_obs[i], rollout_one_post[i, :] = rollout_one.run()
        # rollout_two_obs[i], rollout_two_post[i, :] = rollout_two.run()
        # rollout_three_obs[i], rollout_three_post[i, :] = rollout_three.run()
        # rollout_four_obs[i], rollout_four_post[i, :] = rollout_four.run()
        # rollout_full_obs[i], rollout_full_post[i, :] = rollout_full.run()

    # calculate mean posterior probability of true hypothesis
    active_post_mean = np.mean(active_post, axis=0)
    random_post_mean = np.mean(random_post, axis=0)
    teacher_post_mean = np.mean(teacher_post, axis=0)
    self_teacher_post_mean = np.mean(self_teacher_post, axis=0)
    bayesian_learner_post_mean = np.mean(bayesian_learner_post, axis=0)
    # rollout_none_post_mean = np.mean(rollout_none_post, axis=0)
    # rollout_one_post_mean = np.mean(rollout_one_post, axis=0)
    # rollout_two_post_mean = np.mean(rollout_two_post, axis=0)
    # rollout_three_post_mean = np.mean(rollout_three_post, axis=0)
    # rollout_four_post_mean = np.mean(rollout_four_post, axis=0)
    # rollout_full_post_mean = np.mean(rollout_full_post, axis=0)

    features = np.arange(n_features + 1)

    # plot each run individually
    # for i in range(n_iters):
    #     plt.plot(features, self_teacher_post[i, :], '-ro', alpha=0.01)

    # plt.plot(features, self_teacher_post_mean, '-bo')
    # plt.show()

    plt.plot(features, active_post_mean, '-o', label='Active Learner')
    plt.plot(features, random_post_mean, '-o', label='Random Learner')
    plt.plot(features, teacher_post_mean, '-o', label='Teaching')
    plt.plot(features, self_teacher_post_mean, '-o', label='Self-teaching')
    plt.plot(features, bayesian_learner_post_mean, '-o', label='Weak sampling')
    # plt.plot(features, rollout_none_post_mean, '-o', label='Rollout None')
    # plt.plot(features, rollout_full_post_mean, '-o', label='Rollout Full')
    # plt.plot(features, rollout_one_post_mean, '-o', label='Rollout One')
    # plt.plot(features, rollout_two_post_mean, '-o', label='Rollout Two')
    # plt.plot(features, rollout_three_post_mean, '-o', label="Rollout Three")
    # plt.plot(features, rollout_four_post_mean, '-o', label="Rollout four")
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
