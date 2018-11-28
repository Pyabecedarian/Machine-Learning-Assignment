import numpy as np

from ml_ex4.sigmoid import sigmoid


def sigmoidGradient(z):
    """
    SIGMOIDGRADIENT returns the gradient of the sigmoid function
    evaluated at z
    g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
    evaluated at z. This should work regardless if z is a matrix or a
    vector. In particular, if z is a vector or matrix, you should return
    the gradient for each element.

    g = sigmoid(z)

    dg/dz = g(1-g)

    """
    # % ====================== YOUR CODE HERE ======================
    #   Instructions: Compute the gradient of the sigmoid function evaluated at
    #                 each value of z (z can be a matrix, vector or scalar).
    # g = np.zeros(z.shape)

    g = sigmoid(z) * (1 - sigmoid(z))
    return g