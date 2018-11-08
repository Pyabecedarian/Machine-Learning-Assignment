import numpy as np


def computeCost(X, y, theta):
    """
    COMPUTECOST Compute cost for linear regression
    J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
    parameter for linear regression to fit the data points in X and y

    NOTE:
              x0   x1
            [ 1     x1(1) ]        [  y(1) ]             [ theta0 ]
        X = [ 1     x1(2) ]    y = [  y(2) ]     theta = [ theta1 ]
            [  ...  ...   ]        [  ...  ]

        X.shape = (97, 2)   y.shape = (97,)   theta.shape = (2,)
    """
    m = y.size  # number of training Set

    J = np.sum((theta @ X.T - y) ** 2) / (2 * m)

    return J
