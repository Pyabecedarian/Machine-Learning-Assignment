import numpy as np


def gaussianKernel(x1, x2, sigma):
    """
    RBFKERNEL returns a radial basis function kernel between x1 and x2
    sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
    and returns the value in sim
    :param x1:
    :param x2:
    :param sigma:
    :return:
    """

    # % Ensure that x1 and x2 are column vectors
    x1 = x1.flatten()
    x2 = x2.flatten()

    # % You need to return the following variables correctly.
    sim = 0

    # % ====================== YOUR CODE HERE ======================
    # % Instructions: Fill in this function to return the similarity between x1
    # %               and x2 computed using a Gaussian kernel with bandwidth
    # %               sigma
    # %
    gamma = 1 / (2 * sigma ** 2)
    sim = np.exp(-gamma * (x1 - x2).T @ (x1 - x2))
    # % =============================================================
    return sim