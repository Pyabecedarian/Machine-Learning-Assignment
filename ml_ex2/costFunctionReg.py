import numpy as np
from ml_ex2.sigmoid import sigmoid


def costFunctionReg(theta, X, y, lambd):
    """
    COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
    J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
    theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.
    """

    # % Initialize some useful values
    m = y.size  # % number of training examples

    # % You need to return the following variables correctly
    J = 0
    grad = np.zeros(theta.size)

    # % ====================== YOUR CODE HERE ======================
    # % Instructions: Compute the cost of a particular choice of theta.
    # %               You should set J to the cost.
    # %               Compute the partial derivatives and set grad to the partial
    # %               derivatives of the cost w.r.t. each parameter in theta
    #
    # NOTE:
    #     X.shape = (m, 28)  y.shape = (m, )  theta.shape = (28, )
    #     z.shape = (m, ) = h.shape
    #     J.shape = ()
    #     J(θ) = -1/m * Sum[ -yi*log(h(xi)) - (1-yi)*log(1-h(xi)) ] + λ/2m * Sum[θj^2], for all i and, all j expect j=0
    #     ∂J/∂θ = 1/m * Sum[(h(xi) - yi)*]xji] = X.T @ (h - y) / m
    z = X @ theta
    h = sigmoid(z)
    regtheta = np.hstack((0, theta[1:]))  # Reconstruct a new theta vector with fist element 0 for regularization
    J = (-y@np.log(h) - (1-y)@np.log(1-h)) / m + lambd * (regtheta @ regtheta) / (2*m)
    grad = X.T@(h - y) / m + (lambd/m) * regtheta

    return J, grad
    # % =============================================================
