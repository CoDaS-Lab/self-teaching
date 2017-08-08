import numpy as np
import matplotlib.pyplot as plt

from models.active_learner import ActiveLearner
from models.random_learner import RandomLearner
from models.teacher import Teacher
from models.self_teacher import SelfTeacher
from models.bayesian_learner import BayesianLearner

if __name__ == "__main__":

    n_features = 8
    features = np.arange(n_features)
    n_iters = 1000
    hyp_space_type = "boundary"
    true_hyp_one = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    true_hyp_one = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    # instantiate models
    active_learner = ActiveLearner(n_features, hyp_space_type, true_hyp_one)
    random_learner = RandomLearner(n_features, hyp_space_type, true_hyp_one)
    teacher = Teacher(n_features, hyp_space_type, true_hyp_one)
    self_teacher = SelfTeacher(n_features, hyp_space_type, true_hyp_one)
    bayesian_learner = BayesianLearner(
        n_features, hyp_space_type, true_hyp_one)

    _, _, al_prob = active_learner.run()
    _, _, rl_prob = random_learner.run()
    _, _, t_prob = teacher.run()
    _, _, st_prob = self_teacher.run()
    _, _, bl_prob = bayesian_learner.run()

    # t_prob = t_prob[0]
    # print(t_prob)

    plt.plot(features, al_prob, '-o', label='Active Learner')
    plt.plot(features, rl_prob, '-o', label='Random Learner')
    plt.plot(features, t_prob, '-o', label='Teaching')
    plt.plot(features, st_prob,
             '-o', label='Self-teaching')
    plt.plot(features, bl_prob,
             '-o', label='Weak sampling')
    plt.xlabel("Feature")
    plt.ylabel("Probability of selecting feature")

    plt.legend()
    plt.show()

    # instantiate models
    # active_learner = ActiveLearner(n_features, hyp_space_type, true_hyp_two)
    # random_learner = RandomLearner(n_features, hyp_space_type, true_hyp_two)
    # teacher = Teacher(n_features, hyp_space_type, true_hyp_two)
    # self_teacher = SelfTeacher(n_features, hyp_space_type, true_hyp_two)
    # bayesian_learner = BayesianLearner(
    #     n_features, hyp_space_type, true_hyp_two)

    # _, _, al_prob = active_learner.run()
    # _, _, rl_prob = random_learner.run()
    # _, _, t_prob = teacher.run()
    # _, _, st_prob = self_teacher.run()
    # _, _, bl_prob = bayesian_learner.run()

    # t_prob = t_prob[0]

    # plt.plot(features, al_prob, '-o', label='Active Learner')
    # plt.plot(features, rl_prob, '-o', label='Random Learner')
    # plt.plot(features, t_prob, '-o', label='Teaching')
    # plt.plot(features, st_prob,
    #          '-o', label='Self-teaching')
    # plt.plot(features, bl_prob,
    #          '-o', label='Weak sampling')
    # plt.xlabel("Feature")
    # plt.ylabel("Probability of selecting feature")
    # plt.ylim(0, 1)

    # plt.legend()
    # plt.show()
