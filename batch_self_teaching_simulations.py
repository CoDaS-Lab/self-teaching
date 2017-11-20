import numpy as np
import matplotlib.pyplot as plt

from models.batch_self_teacher import BatchSelfTeacher

import warnings
warnings.filterwarnings('ignore')

# run batch self teaching code
n_features = 8
n_iters = 1000
batch_size = 1

batch_self_teacher_boundary_obs = np.zeros(n_iters)
batch_self_teacher_boundary_post = np.zeros((n_iters, n_features + 1))
active_learner_boundary_obs = np.zeros(n_iters)
active_learner_boundary_post = np.zeros((n_iters, n_features + 1))

# run boundary simulations
hyp_space_type = "boundary"
sampling = "max"
for i in range(n_iters):
    if i % 100 == 0:
        print(i)
    batch_self_teacher = BatchSelfTeacher(n_features, hyp_space_type, batch_size, sampling=sampling)
    batch_self_teacher_boundary_obs[i], batch_self_teacher_boundary_post[i,
                                                             :], batch_self_teacher_prob = batch_self_teacher.run()
    active_learner = ActiveLearner(n_features, hyp_space_type, sampling=sampling)
    active_learner_boundary_obs[i], active_learner_boundary_post[i,
                                                             :], active_learner_prob = active_learner.run()
print("done")

# plot results
batch_self_teacher_boundary_post_mean = np.mean(batch_self_teacher_boundary_post, axis=0)
active_learner_boundary_post_mean = np.mean(active_learner_boundary_post, axis=0)
plt.plot(np.arange(len(batch_self_teacher_boundary_post_mean)), batch_self_teacher_boundary_post_mean,
         label = "batch self teacher")
plt.plot(np.arange(len(active_learner_boundary_post_mean)), active_learner_boundary_post_mean,
         label = "active learner")
plt.legend()
plt.show()
