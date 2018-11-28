import numpy as np
import scipy.io as sio

from ml_ex4.sigmoid import sigmoid
from ml_ex4.sigmoidGradient import sigmoidGradient


def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lmbd):
    """
    NNCOSTFUNCTION Implements the neural network cost function for a two layer
    neural network which performs classification
    [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
    X, y, lambda) computes the cost and gradient of the neural network. The
    parameters for the neural network are "unrolled" into the vector
    nn_params and need to be converted back into the weight matrices.

    The returned parameter grad should be a "unrolled" vector of the
    partial derivatives of the neural network.

    NOTE:
        Theta1.shape = (25, 401)   Theta2.shape = (10, 26)
    """
    # % Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    # % for our 2 layer neural network
    indices1 = hidden_layer_size * (input_layer_size + 1)
    Theta1 = nn_params[:indices1].reshape(hidden_layer_size, input_layer_size + 1, order='F')
    Theta2 = nn_params[indices1:].reshape(num_labels, hidden_layer_size + 1, order='F')

    # % Setup some useful variables
    m, _ = X.shape
    # % You need to return the following variables correctly
    J = 0
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)

    #     % ====================== YOUR CODE HERE ======================
    # % Instructions: You should complete the code by working through the
    # %               following parts.
    # %
    # % Part 1: Feedforward the neural network and return the cost in the
    # %         variable J. After implementing Part 1, you can verify that your
    # %         cost function computation is correct by verifying the cost
    # %         computed in ex4.m
    # %
    # % Part 2: Implement the backpropagation algorithm to compute the gradients
    # %         Theta1_grad and Theta2_grad. You should return the partial derivatives of
    # %         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
    # %         Theta2_grad, respectively. After implementing Part 2, you can check
    # %         that your implementation is correct by running checkNNGradients
    # %
    # %         Note: The vector y passed into the function is a vector of labels
    # %               containing values from 1..K. You need to map this vector into a
    # %               binary vector of 1's and 0's to be used with the neural network
    # %               cost function.
    # %
    # %         Hint: We recommend implementing backpropagation using a for-loop
    # %               over the training examples if you are implementing it for the
    # %               first time.
    # %
    # % Part 3: Implement regularization with the cost function and gradients.
    # %
    # %         Hint: You can implement this around the code for
    # %               backpropagation. That is, you can compute the gradients for
    # %               the regularization separately and then add them to Theta1_grad
    # %               and Theta2_grad from Part 2.
    # %
    """
     J(θ) = 1/m * Σm Σk [-y*log(h(x)) - (1-y)*log(1-h)]
     
     
     Compute h:
     ------------------------------------------------------------------------------------------------
         Layer1                                  Layer2                                      Layer3
         X(5000, 400)
         add 1's         θ1(25, 401)
     (5000, 401) ------- g( X@θ1.T )--------  a2=(5000,25)
                                                 add 1's            θ2(10,26)
                                                (5000, 26)--------g( a2@θ2.T )---------   a3(5000,10)
     ------------------------------------------------------------------------------------------------
     For each xi, xi.shape = (401,),  yi's shape should be yi.shape=(10,),  thus
     if X.shape = (m,401), then
        y.shape = (m, 10)
                                             
    """
    Y = np.zeros((m, num_labels))    # Y.shape = (m, 10)
    for i in range(m):
        Y[i, y[i]-1] = 1             # eg. yi = 10  --->  Yi = [0,0,0,0,0,0,0,0,0,1]

    # Compute h(x)
    # Layer1 -> Layer2
    X = np.hstack((np.ones(m).reshape(m,1), X))    # X.shape = (m, 401)
    a2 = sigmoid(X @ Theta1.T)                     # a2.shape = (m x 401 @ 401 x 25) = (m, 25)
    # Layer2 -> Layer3
    a2 = np.hstack((np.ones(m).reshape(m,1), a2))  # a2.shape = (m, 26)
    h = sigmoid(a2 @ Theta2.T)                     # h.shape = (m x 26 @ 26 x 10) = (m x 10)

    # Unregularized cost J
    J = np.sum(-Y * np.log(h) - (1-Y) * np.log(1-h))/m
    # Regularize J
    reg_theta1 = Theta1[:, 1:]
    reg_theta2 = Theta2[:, 1:]
    reg_term = (lmbd / (2*m)) * (np.sum(reg_theta1 ** 2) + np.sum(reg_theta2 ** 2))
    J += reg_term

    # Compute ∂J/∂θ using backpropagation
    pd_theta2_Cost = np.zeros((num_labels, hidden_layer_size+1))
    pd_theta1_Cost = np.zeros((hidden_layer_size, input_layer_size+1))
    for i in range(m):
        # Preparation:
        theta1 = Theta1         # theta1.shape = (25, 401)
        a1 = X[i]               #     a1.shape = (401,   )
        z2 = theta1 @ a1        #     z2.shape = (25,    )
        a2_temp = sigmoid(z2)
        theta2 = Theta2         # theta2.shape = (10, 26)
        a2 = np.hstack((1, a2_temp))  # a2.shape = (26, )
        z3 = theta2 @ a2              # z3.shape = (10, )
        a3 = sigmoid(z3)
        yt = Y[i]                     # y3.shape = (10, )

        # Compute δl, for l = 3, 2
        delta3 = a3 - yt                                #     δ3.shape = (10, )
        g_prime_z2 = sigmoidGradient(z2)                # g'(z2).shape = (25, )
        delta2 = theta2[:, 1:].T @ delta3 * g_prime_z2  #     δ2.shape = (25, ) = (25, 10) @ (10, )

        # Compute & update partial derivative of C with respect all θ: ▽θ2,  ▽b2,  ▽θ1,  ▽b2
        pd_theta2_Cost_temp = np.outer(delta3, a2)              # ▽θ2.shape = (10, 26 ) = (10, 1) @ "(1, 26 )"
        pd_theta1_Cost_temp = np.outer(delta2, a1)              # ▽θ1.shape = (25, 401) = (25, 1) @ "(1, 401)"

        pd_theta2_Cost += pd_theta2_Cost_temp
        pd_theta1_Cost += pd_theta1_Cost_temp

    # Compute ∂J/∂θ
    # Unregularized Gradient
    pd_theta2_J = pd_theta2_Cost / m
    pd_theta1_J = pd_theta1_Cost / m
    # Regularize Gradient
    pd_theta2_J[:,1:] += lmbd / m * Theta2[:, 1:]
    pd_theta1_J[:,1:] += lmbd / m * Theta1[:, 1:]

    Theta2_grad = pd_theta2_J
    Theta1_grad = pd_theta1_J

    grad = np.concatenate((Theta1_grad.reshape(Theta1_grad.size, order='F'),
                           Theta2_grad.reshape(Theta2_grad.size, order='F')))
    return J, grad


if __name__ == '__main__':
    Data = sio.loadmat('ex4data1.mat')
    X = Data['X']
    y = Data['y'].flatten()

    Theta = sio.loadmat('ex4weights.mat')
    Theta1 = Theta['Theta1']
    Theta2 = Theta['Theta2']

    nn_params = np.concatenate((Theta1.reshape(Theta1.size, order='F'),
                                Theta2.reshape(Theta2.size, order='F')))
    input_layer_size = 400  # % 20x20 Input Images of Digits
    hidden_layer_size = 25  # % 25 hidden units
    num_labels = 10
    Y = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, 0.1)
