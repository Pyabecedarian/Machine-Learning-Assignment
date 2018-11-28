import numpy as np

from ml_ex4.sigmoid import sigmoid
from ml_ex4.sigmoidGradient import sigmoidGradient


def nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lmbd):
    """
    Compute J(θ) and ∂J/∂θ
    :param nn_params: initial_parameters
    :param input_layer_size: 400
    :param hidden_layer_size: 25
    :param num_labels: 10
    :param X:  X.shape = (5000, 400)
    :param y:  y.shape = (5000, )
    :param lmbd:
    :return:
    """
    # 1. ----------------------------- Compute J(θ) -------------------------------
    #      J = 1/m Σm Σk [-yi*log(hi) - (1-y)*log(1-hi)] + λ/2m * Σj=1 Σi=1 (θ^2)
    # Preparations
    m, _ = X.shape

    Y = np.zeros((m, num_labels))
    for i in range(m):
        Y[i, y[i]-1] = 1

    theta1 = nn_params[:hidden_layer_size*(input_layer_size+1)].reshape(
        hidden_layer_size, input_layer_size+1, order='F')  # theta1.shape = (25, 401)
    theta2 = nn_params[hidden_layer_size*(input_layer_size+1):].reshape(
        num_labels, hidden_layer_size+1, order='F')        # theta2.shape = (10, 26)

    a1 = np.c_[np.ones(m), X]                              # a1.shape = (m,  401)
    z2 = a1 @ theta1.T                                     # z2.shape = (m, 25)
    a2 = np.c_[np.ones(m), sigmoid(z2)]                    # a2.shape = (m, 26)
    z3 = a2 @ theta2.T                                     # z3.shape = (m, 10)
    h = a3 = sigmoid(z3)                                   # a3.shape = (m, 10)

    # Unregularized J
    #   (m, 10)  (m, 10)
    J = np.sum(-Y * np.log(h) - (1-Y) * np.log(1-h)) / m
    # Regularize J
    reg_term = lmbd * (np.sum(theta1[:, 1:]**2) + np.sum(theta2[:, 1:]**2)) / (2*m)
    J += reg_term

    # 2. -------------------------- Compute ∂J/∂θ using BP Algorithm -----------------------
    # Preparation
    pd_theta2_cost = np.zeros((num_labels, hidden_layer_size+1))        # ▽θ2(C).shape = (10, 25)
    pd_theta1_cost = np.zeros((hidden_layer_size, input_layer_size+1))  # ▽θ1(C).shape = (25, 401)

    for i in range(m):
        delta3 = a3[i] - Y[i]                                        # δ3.shape = (10, )
        delta2 = theta2[:, 1:].T @ delta3 * sigmoidGradient(z2[i])   # δ2.shape = (25, )
        # Compute Partial Derivative
        pd_theta2_cost[:, 1:] += np.outer(delta3, a2[i, 1:])
        pd_theta1_cost[:, 1:] += np.outer(delta2, a1[i, 1:])
        pd_theta2_cost[:, 0] += delta3
        pd_theta1_cost[:, 0] += delta2

    reg_theta1 = np.c_[np.zeros(hidden_layer_size), theta1[:, 1:]]
    reg_theta2 = np.c_[np.zeros(num_labels), theta2[:, 1:]]
    grad_theta1 = 1/m * pd_theta1_cost + lmbd/m * reg_theta1
    grad_theta2 = 1/m * pd_theta2_cost + lmbd/m * reg_theta2
    grad = np.concatenate((grad_theta1.reshape(grad_theta1.size, order='F'),
                           grad_theta2.reshape(grad_theta2.size, order='F')))

    return J, grad
