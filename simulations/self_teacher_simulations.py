import sys
sys.path.insert(0, '../')
import numpy as np
import matplotlib.pyplot as plt
from concept_learning.self_teacher import SelfTeacher
from concept_learning.active_learner import ActiveLearner
import warnings
warnings.filterwarnings('ignore')

# run self teaching code
n_features = 8
n_iters = 1000

self_teacher_boundary_obs = np.zeros(n_iters)
self_teacher_boundary_post = np.zeros((n_iters, n_features + 1))
active_learner_boundary_obs = np.zeros(n_iters)
active_learner_boundary_post = np.zeros((n_iters, n_features + 1))

# run boundary simulations
hyp_space_type = "boundary"
sampling = "max"

for i in range(n_iters):
    if i % 100 == 0:
        print(i)
    self_teacher = SelfTeacher(n_features, hyp_space_type, sampling=sampling)
    self_teacher_boundary_obs[i], self_teacher_boundary_post[i,
                                                             :], self_teacher_prob = self_teacher.run()
    active_learner = ActiveLearner(n_features, hyp_space_type, sampling=sampling)
    active_learner_boundary_obs[i], active_learner_boundary_post[i,
                                                             :], active_learner_prob = active_learner.run()

# plot results
self_teacher_boundary_post_mean = np.mean(self_teacher_boundary_post, axis=0)
active_learner_boundary_post_mean = np.mean(active_learner_boundary_post, axis=0)
plt.plot(np.arange(len(self_teacher_boundary_post_mean)), self_teacher_boundary_post_mean,
         label = "self teacher")
plt.plot(np.arange(len(active_learner_boundary_post_mean)), active_learner_boundary_post_mean,
         label = "active learner")
plt.legend()
plt.show()


# f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
# ax1.plot(np.arange(len(self_teacher_boundary_post_mean)), self_teacher_boundary_post_mean)
# ax1.set_title('Self Teacher')
# ax2.plot(np.arange(len(active_learner_boundary_post_mean)), active_learner_boundary_post_mean)
# ax2.set_title('Active Learning')
# plt.show()


# plot first feature prob
plt.plot(np.arange(n_features), self_teacher_prob ** 3 / np.sum(self_teacher_prob ** 3), label = "self teacher")
plt.plot(np.arange(n_features), active_learner_prob, label = "active learner")
plt.legend()
plt.show()
