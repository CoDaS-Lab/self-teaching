from EIG.utils import read_file
import numpy as np

class Eig:

    ## refract notes: call super to init
    def __init__(self, hyp_space_fpath):

        self.hyp_space = read_file(hyp_space_fpath)
        self.num_features = self.hyp_space.shape[1]
        self.num_hyp = self.hyp_space.shape[0]
        self.n = self.hyp_space.shape[0]


        ## hyp_space: assumes that space is created elsewhere and passed in as file...(maybe refactor)

    def likelihood(self):

        return np.log2(self.n)

    def update_learner_posterior(self, n_yes, n_no):

        return ((n_no/self.n)*(np.log2(n_no))) + ((n_yes/self.n) * (np.log2(n_yes)))

    def entropy(self):
        pass

    def inf_gain(self, feature_idx):

        n_yes = (self.hyp_space[:, feature_idx] == 1).sum()
        n_no = (self.hyp_space[:, feature_idx] == 0).sum()

        if n_yes == 0:
            return 0

        return self.likelihood() - self.update_learner_posterior(n_yes, n_no)



