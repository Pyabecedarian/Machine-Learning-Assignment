import numpy as np


def sigmoid(Z):
    g = 1 / (1 + np.exp(-Z))

    return g
