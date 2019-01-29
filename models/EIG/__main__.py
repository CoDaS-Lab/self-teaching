import os, sys
from EIG.eig import Eig


def main():

    #* First block handles file i/o exceptions if any *#
    # refactor so fpath is passed through argv[] #
    if not os.path.isfile('non_representative_features.txt'):
        print("file does not exist. exiting...")
        sys.exit(2)


    # Instantiate class objects for both representative and non-rep test cases (Markant, "Guess Who") #
    non_rep = Eig('non_representative_features.txt')
    rep = Eig('representative_features.txt')

    # Print IG values for both test cases (sanity check) #
    for i in range(rep.n_features):
        print(i+ 1, rep.inf_gain(i))

    print('*****************************')

    for i in range(non_rep.n_features):
        print(i+1, non_rep.inf_gain(i))



if __name__ == '__main__':
    main()