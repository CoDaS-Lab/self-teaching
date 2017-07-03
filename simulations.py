import numpy as np
import matplotlib.pyplot as plt

from models.active_learner import ActiveLearner
from models.random_learner import RandomLearner
from models.teacher import Teacher

if __name__ == "__main__":
    n_features = 8
    n_iters = 1000

    # save number of observations for each iteration
    active_obs = np.zeros(n_iters)
    random_obs = np.zeros(n_iters)
    teacher_obs = np.zeros(n_iters)

    # save posterior probability for each iteration
    active_post = np.zeros((n_iters, n_features))
    random_post = np.zeros((n_iters, n_features))
    teacher_post = np.zeros((n_iters, n_features))

    for i in range(n_iters):
        # create active, teacher and random learners
        active_learner = ActiveLearner(n_features)
        random_learner = RandomLearner(n_features)
        teacher = Teacher(n_features)

        # run simulations across all three learners
        active_obs[i], active_post[i, :] = active_learner.run()
        random_obs[i], random_post[i, :] = random_learner.run()
        teacher_obs[i], teacher_post[i, :] = teacher.run()

    features = np.arange(n_features + 1)
    active_learner_counts = np.bincount(active_obs.astype(int)) / n_iters
    random_learner_counts = np.bincount(random_obs.astype(int)) / n_iters
    teacher_counts = np.bincount(teacher_obs.astype(int)) / n_iters

    plt.bar(features, active_learner_counts, 0.2)
    plt.bar(features + 0.2, random_learner_counts, 0.2)
    plt.bar(features + 0.4, teacher_counts, 0.2)
    plt.show()
