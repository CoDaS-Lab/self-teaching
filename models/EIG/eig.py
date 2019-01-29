from EIG.utils import read_file
import numpy as np

class Eig:

    ## refract notes: call super to init
    def __init__(self, hyp_space_fpath, n_label=2):

        self.hyp_space = read_file(hyp_space_fpath)
        self.n_features = self.hyp_space.shape[1]
        self.n_hyp = self.hyp_space.shape[0]
        self.n = self.hyp_space.shape[0]

        self.prior = 1 / self.n_hyp * np.ones((self.n_hyp, self.n_features, n_label))
        ## hyp_space: assumes that space is created elsewhere and passed in as file...(maybe refactor)

    def prior_ent(self):

        return np.log2(self.n)

    def update_learner_posterior(self, n_yes, n_no):

        dist = np.sum(self.posterior, axis=0)

        self.posterior = np.nan_to_num(np.divide(self.posterior, dist, where=dist!=0))

    def posterior_ent(self, n_yes, n_no):

        return ((n_no / self.n) * (np.log2(n_no))) + ((n_yes / self.n) * (np.log2(n_yes)))

    def entropy(self):

        return np.nansum(self.prior * (1 / np.log2(self.prior)), axis=0)

    def inf_gain(self, feature_idx):

        n_yes = (self.hyp_space[:, feature_idx] == 1).sum()
        n_no = (self.hyp_space[:, feature_idx] == 0).sum()

        if n_yes == 0:
            return 0

        return self.prior_ent() - self.posterior_ent(n_yes, n_no)



