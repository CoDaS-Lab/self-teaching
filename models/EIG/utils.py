import os, numpy as np


def read_file(fpath, ext='.txt'):

    # later variants will consider other file types
    if ext == '.txt':
        return np.asarray(np.loadtxt(fpath))

