import numpy as np
from numpy.linalg import pinv


def normalEqn(X, y):
    """
    NORMALEQN Computes the closed-form solution to linear regression
    NORMALEQN(X,y) computes the closed-form solution to linear
    regression using the normal equations.
    :param X:   X.shape = (47, 3)
    :param y:   y.shape = (47, )
    :return: theta
    """

    theta = np.zeros(X[1].size)
    # % ====================== YOUR CODE HERE ======================
    # % Instructions: Complete the code to compute the closed form solution
    # %               to linear regression and put the result in theta.

    theta = pinv(X.T @ X) @ X.T @ y
    # % ---------------------- Sample Solution ----------------------
    #
    # % -------------------------------------------------------------
    return theta

    # % ============================================================
