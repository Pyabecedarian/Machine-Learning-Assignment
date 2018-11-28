from ml_ex4.computeNumericalGradient import computeNumericalGradient
from ml_ex4.debugInitializeWeights import debugInitializeWeights
from ml_ex4.nnCostFunction import nnCostFunction
import numpy as np

from ml_ex4.nn_cost_function import nn_cost_function


def checkNNGradients(lmbd=None):
    """
    CHECKNNGRADIENTS Creates a small neural network to check the
    backpropagation gradients
    CHECKNNGRADIENTS(lmbd)
    backpropagation gradients, it will output the analytical gradients
    produced by your backprop code and the numerical gradients (computed
    using computeNumericalGradient). These two gradient computations should
    result in very similar values.
    """
    if lmbd is None:
      lmbd = 0

    input_layer_size = 3
    hidden_layer_size = 5
    num_lables = 3
    m = 5

    # % We generate some 'random' test data
    Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
    Theta2 = debugInitializeWeights(num_lables, hidden_layer_size)
    # % Reusing debugInitializeWeights to generate X
    X = debugInitializeWeights(m, input_layer_size - 1)
    y = 1 + np.mod(np.arange(1, m+1), num_lables)

    # % Unroll parameters
    nn_params = np.concatenate((Theta1.reshape(Theta1.size, order='F'), Theta2.reshape(Theta2.size, order='F')))

    # % Short hand for cost function
    def cost_func(p):
        # return nnCostFunction(p, input_layer_size, hidden_layer_size, num_lables, X, y, lmbd)
        return nn_cost_function(p, input_layer_size, hidden_layer_size, num_lables, X, y, lmbd)

    cost, grad = cost_func(nn_params)
    numgrad = computeNumericalGradient(cost_func, nn_params)

    # % Visually examine the two gradient computations.  The two columns
    # % you get should be very similar.
    print(np.c_[numgrad, grad])
    print('The above two columns you get should be very similar.\n'
          '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n')

    # Evaluate the norm of the difference between two solutions.
    # If you have a correct implementation, and assuming you used EPSILON = 0.0001
    # in computeNumericalGradient.m, then diff below should be less than 1e-9
    diff = np.linalg.norm(numgrad-grad) / np.linalg.norm(numgrad+grad)
    print('If your backpropagation implementation is correct, then \n'
          'the relative difference will be small (less than 1e-9). \n'
          'Relative Difference: %g\n' % diff)