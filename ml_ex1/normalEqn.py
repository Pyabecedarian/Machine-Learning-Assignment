import numpy as np


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
    #
    # % ====================== YOUR CODE HERE ======================
    # % Instructions: Complete the code to compute the closed form solution
    # %               to linear regression and put the result in theta.
    # %
    from numpy.linalg import pinv
    theta = pinv(X.T @ X) @ X.T @ y
    # % ---------------------- Sample Solution ----------------------
    #
    #
    #
    #
    # % -------------------------------------------------------------
    #
    return theta
#
# % ============================================================


if __name__ == '__main__':
    # Data3 = np.loadtxt('ml_ex1/ex1data2.txt', delimiter=',')
    Data3 = np.loadtxt('ex1data2.txt', delimiter=',')
    X = Data3[:,:2]
    y = Data3[:, 2]
    m = y.size

    X = np.hstack((np.ones(m).reshape(m, 1), X))
    print(X.shape, y.shape)

    theta = normalEqn(X, y)
    print(theta)