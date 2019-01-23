import numpy as np
import os, sys


from EIG.eig import Eig

def main():

    # refactor so fpath is passed through argv[] #
    if not os.path.isfile('non_representative_features.txt'):
        print("file does not exist. exiting...")
        sys.exit(2)
        
    non_rep = Eig('non_representative_features.txt')
    rep = Eig('representative_features.txt')

    for i in range(rep.num_features):
        print(i+ 1, rep.inf_gain(i))

    print('*****************************')

    for i in range(non_rep.num_features):
        print(i+1, non_rep.inf_gain(i))







if __name__ == '__main__':
    main()