from EIG.utils import read_file

class Eig:

    ## refract notes: call super to init
    def __init__(self, hyp_space_fpath):

        self.hyp_space = read_file(hyp_space_fpath)
        print(self.hyp_space)
        self.num_features = self.hyp_space[1]
        self.num_hyp = self.hyp_space[0]

        ## hyp_space: assumes that space is created elsewhere and passed in as file...(maybe refactor)




    def likelihood(self):
        pass

    def update_learner_posterior(self):
        pass

    def entropy(self):
        pass

    def inf_gain(self):
        pass