import numpy as np


def linearRegCostFunction(X, y, theta, lmbd):
    """
    LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
    regression with multiple variables
      [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
      cost of using theta as the parameter for linear regression to fit the
      data points in X and y. Returns the cost in J and the gradient in grad
    :param X:  X.shape = (12, 2)
    :param y:  y.shape = (12, 1)
    :param theta:  θ.shape = (2, )
    :param lmbd:
    :return:
    """

    # % Initialize some useful values
    m = y.shape[0]  # % number of training examples

    # % You need to return the following variables correctly
    J = 0
    grad = np.zeros(theta.size)

    # % ====================== YOUR CODE HERE ======================
    # % Instructions: Compute the cost and gradient of regularized linear
    # %               regression for a particular choice of theta.
    # %
    # %               You should set J to the cost and grad to the gradient.
    # %
    # NOTE:
    #      J(θ) = 1/m * Σm (hi - yi)^2 + λ/2m * Σi=1 (θ^2)
    #      h(x) = θ0 + θ1*x
    # 1. ------------------------ Compute J(θ) ----------------------------------
    # Unregularized J
    h = X @ theta            # h.shape = (12, )
    J = (h - y) @ (h - y) / (2*m)
    # Regularize J
    reg_term_J = lmbd * np.sum(theta[1:]**2) / (2*m)
    J += reg_term_J

    # 2. -------------------------- Compute ∂J/∂θ -------------------------------
    # Unregularized grad
    grad = (h - y) @ X / m
    # Regularize grad
    reg_term_theta = lmbd * np.hstack((0, theta[1:])) / m
    grad += reg_term_theta

    # % =========================================================================
    return J, grad