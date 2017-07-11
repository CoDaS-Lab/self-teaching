import numpy as np
import matplotlib.pyplot as plt

from models.active_learner import ActiveLearner
from models.random_learner import RandomLearner
from models.teacher import Teacher
from models.self_teacher import SelfTeacher

if __name__ == "__main__":
    n_features = 10
    n_iters = 1000

    # save number of observations for each iteration
    active_obs = np.zeros(n_iters)
    random_obs = np.zeros(n_iters)
    teacher_obs = np.zeros(n_iters)
    self_teacher_obs = np.zeros(n_iters)

    # save posterior probability for each iteration
    active_post = np.zeros((n_iters, n_features + 1))
    random_post = np.zeros((n_iters, n_features + 1))
    teacher_post = np.zeros((n_iters, n_features + 1))
    self_teacher_post = np.zeros((n_iters, n_features + 1))

    for i in range(n_iters):
        # create active, teacher and random learners
        active_learner = ActiveLearner(n_features)
        random_learner = RandomLearner(n_features)
        teacher = Teacher(n_features)
        self_teacher = SelfTeacher(n_features)

        # run simulations across all three learners
        active_obs[i], active_post[i, :] = active_learner.run()
        random_obs[i], random_post[i, :] = random_learner.run()
        teacher_obs[i], teacher_post[i, :] = teacher.run()
        self_teacher_obs[i], self_teacher_post[i, :] = self_teacher.run()

    # calculate mean posterior probability of true hypothesis
    active_post_mean = np.mean(active_post, axis=0)
    random_post_mean = np.mean(random_post, axis=0)
    teacher_post_mean = np.mean(teacher_post, axis=0)
    self_teacher_post_mean = np.mean(self_teacher_post, axis=0)

    features = np.arange(n_features + 1)

    # for i in range(n_iters):
    #     plt.plot(features, self_teacher_post[i, :], '-ro', alpha=0.01)

    # plt.plot(features, self_teacher_post_mean, '-bo')
    # plt.show()

    plt.plot(features, active_post_mean, '-og', label='Active Learner')
    plt.plot(features, random_post_mean, '-or', label='Random Learner')
    plt.plot(features, teacher_post_mean, '-ob', label='Teaching')
    plt.plot(features, self_teacher_post_mean, '-ok', label='Self-teaching')
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