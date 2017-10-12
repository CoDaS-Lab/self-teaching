import numpy as np
import matplotlib.pyplot as plt

from models.self_teacher import SelfTeacher
from models.active_learner import ActiveLearner

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
for i in range(n_iters):
    print(i)
    self_teacher = SelfTeacher(n_features, hyp_space_type)
    self_teacher_boundary_obs[i], self_teacher_boundary_post[i,
                                                             :], _ = self_teacher.run()
    active_learner = ActiveLearner(n_features, hyp_space_type)
    active_learner_boundary_obs[i], active_learner_boundary_post[i,
                                                             :], _ = active_learner.run()


# plot results
self_teacher_boundary_post_mean = np.mean(self_teacher_boundary_post, axis=0)
active_learner_boundary_post_mean = np.mean(self_teacher_boundary_post, axis=0)
plt.plot(np.arange(len(self_teacher_boundary_post_mean)), self_teacher_boundary_post_mean)
plt.plot(np.arange(len(active_learner_boundary_post_mean)), active_learner_boundary_post_mean)
plt.show()


f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot(np.arange(len(self_teacher_boundary_post_mean)), self_teacher_boundary_post_mean)
ax1.set_title('Self Teacher')
ax2.plot(np.arange(len(active_learner_boundary_post_mean)), active_learner_boundary_post_mean)
ax2.set_title('Active Learning')
plt.show()
