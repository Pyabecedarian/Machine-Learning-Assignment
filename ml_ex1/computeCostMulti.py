import numpy as np


def computeCostMulti(X, y, theta):
    """
    %COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
    J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
    parameter for linear regression to fit the data points in X and y

    :param X:  X.shape = (47, 3)
    :param y:  y.shape = (47, )
    :param theta: vector [theta0, theta1, theta2], theta.shape = (3, )
    :return: J:  Cost Function
    """
    # Initialize some useful values
    m =y.size  # number of training examples

    # % You need to return the following variables correctly
    # J = 0

    # % ====================== YOUR CODE HERE ======================
    # % Instructions: Compute the cost of a particular choice of theta
    # %               You should set J to the cost.
    # J = np.sum((theta @ X.T - y)) / (2*m)
    A = theta @ X.T - y
    J = (A.T @ A) / (2*m)

    return J
# % =========================================================================
if __name__ == '__main__':

    Data2 = np.loadtxt('ex1data2.txt', delimiter=',')
    X = Data2[:, :2]
    y = Data2[:, 2]
    m = y.size
    X = np.hstack((np.ones(m).reshape(m, 1), X))

    J = computeCostMulti(X, y, theta=np.array([0,0,0]))
    print(J)
