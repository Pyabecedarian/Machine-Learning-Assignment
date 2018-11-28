import numpy as np
from ml_ex3.sigmoid import sigmoid
import scipy.io as sio


def lrCostFunction(theta, X, y, lmbd):
    """
    LRCOSTFUNCTION Compute cost and gradient for logistic regression with
    regularization
    J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
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
    # %
    # % Hint: The computation of the cost function and gradients can be
    # %       efficiently vectorized. For example, consider the computation
    # %
    # %           sigmoid(X * theta)
    # %
    # %       Each row of the resulting matrix will contain the value of the
    # %       prediction for that example. You can make use of this to vectorize
    # %       the cost function and gradient computations.
    # %
    # % Hint: When computing the gradient of the regularized cost function,
    # %       there're many possible vectorized solutions, but one solution
    # %       looks like:
    # %           grad = (unregularized gradient for logistic regression)
    # %           temp = theta;
    # %           temp(1) = 0;   % because we don't add anything for j = 0
    # %           grad = grad + YOUR_CODE_HERE (using the temp variable)
    # %
    h = sigmoid(X @ theta)
    regTheta = np.hstack((0, theta[1:]))

    # Unregularized cost ---- J
    J_unreg = (-y @ np.log(h) - (1 - y) @ np.log(1 - h)) / m
    # Regularize J
    J = J_unreg + lmbd * (regTheta @ regTheta) / (2 * m)

    # Unregularized gradient ----  ∂J/∂θ
    grad_unreg = X.T @ (h - y) / m  # Also:  grad_unreg =  (h - y) @ X / m
    # Regularize grad
    grad = grad_unreg + lmbd * regTheta / m

    # % =============================================================
    return J, grad


if __name__ == '__main__':
    data_dict = sio.loadmat('ex3data1.mat')  # % training data will be stored in arrays X, y
    X = data_dict['X']
    y = data_dict['y'].flatten()
    theta_t = np.array([-2, -1, 1, 2])
    X_t = np.hstack((np.ones(5).reshape(5, 1), np.arange(1, 16).reshape(3, 5).T / 10))
    y_t = 1 * (np.array([1, 0, 1, 0, 1]) >= 0.5)
    lambda_t = 3
    J, grad = lrCostFunction(theta_t, X_t, y_t, lambda_t)
    print(J)
    print(grad)