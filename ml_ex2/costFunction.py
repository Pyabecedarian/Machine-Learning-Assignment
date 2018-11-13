import numpy as np
from ml_ex2.sigmoid import sigmoid


def costFunction(theta, X, y):
    """
    %COSTFUNCTION Compute cost and gradient for logistic regression
    J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
    parameter for logistic regression and the gradient of the cost
    w.r.t. to the parameters.

    :param theta:  theta.shape = (3, )
    :param X:          X.shape = (100, 3)
    :param y:          Y.shape = (100, )
    :return:  J, grad
    """

    # % Initialize some useful values
    m = y.size  # % number of training examples

    # % You need to return the following variables correctly
    # J = 0
    # grad = np.zeros(theta.size)

    # % ====================== YOUR CODE HERE ======================
    # % Instructions: Compute the cost of a particular choice of theta.
    # %               You should set J to the cost.
    # %               Compute the partial derivatives and set grad to the partial
    # %               derivatives of the cost w.r.t. each parameter in theta
    # %
    # % Note: grad should have the same dimensions as theta
    # % Recall:  J(θ) = -1/m * Sum[ -yi*log(h(xi)) - (1-yi)*log(1-h(xi)) ]
    # %
    z = X @ theta
    h = sigmoid(z)  # Hypothesis function --- h(x) = g(z), where z = θ.T @ X
    J = (y @ np.log(h) + (1 - y) @ np.log(1 - h)) / (-m)  # Cost function --- J(θ)
    grad = X.T @ (h - y) / m  # Partial derivatives

    return J, grad
    # % =============================================================
