"""
%% Machine Learning Online Class - Exercise 4 Neural Network Learning

%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the
%  linear exercise. You will need to complete the following functions
%  in this exericse:
%
%     sigmoidGradient.m
%     randInitializeWeights.m
%     nnCostFunction.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%
"""
import scipy.io as sio
import numpy as np
import scipy.optimize as opt
import time

from ml_ex4.checkNNGradients import checkNNGradients
from ml_ex4.displayData import displayData
from ml_ex4.nn_cost_function import nn_cost_function
from ml_ex4.predict import predict
from ml_ex4.randInitializeWeights import randInitializeWeights
from ml_ex4.sigmoidGradient import sigmoidGradient


# %% Setup the parameters you will use for this exercise
input_layer_size = 400  # % 20x20 Input Images of Digits
hidden_layer_size = 25  # % 25 hidden units
num_labels = 10         # % 10 labels, from 1 to 10
                        # % (note that we have mapped "0" to label 10)


print('p1 -----------------------------------------------------')
# %% =========== Part 1: Loading and Visualizing Data =============
# %  We start the exercise by first loading and visualizing the dataset.
# %  You will be working with a dataset that contains handwritten digits.
# %
# % Load Training Data
print('Loading and Visualizing Data ...\n')
Data = sio.loadmat('ex4data1.mat')
X = Data['X']
y = Data['y'].flatten()

# % Randomly select 100 data points to display
sel = np.random.permutation(X)
sel = sel[:100, :]
displayData(sel)
print('Program paused. Press enter to continue.\n')


print('p2 -----------------------------------------------------')
# %% ================ Part 2: Loading Parameters ================
# % In this part of the exercise, we load some pre-initialized
# % neural network parameters.
print('\nLoading Saved Neural Network Parameters ...\n')
# % Load the weights into variables Theta1 and Theta2
Theta = sio.loadmat('ex4weights.mat')
Theta1 = Theta['Theta1']
Theta2 = Theta['Theta2']

# % Unroll parameters
# NOTE: In Matlab, the elements are unraveled first down the columns, then by the rows. In Python it's the opposite.
#       A(:) is equivalent to A.reshape(A.size, order='F'),  where 'F' refers to Fortran order
nn_params = np.concatenate((Theta1.reshape(Theta1.size, order='F'), Theta2.reshape(Theta2.size, order='F')))


print('p3 -----------------------------------------------------')
# %% ================ Part 3: Compute Cost (Feedforward) ================
# %  To the neural network, you should first start by implementing the
# %  feedforward part of the neural network that returns the cost only. You
# %  should complete the code in nnCostFunction.m to return cost. After
# %  implementing the feedforward to compute the cost, you can verify that
# %  your implementation is correct by verifying that you get the same cost
# %  as us for the fixed debugging parameters.
# %
# %  We suggest implementing the feedforward cost *without* regularization
# %  first so that it will be easier for you to debug. Later, in part 4, you
# %  will get to implement the regularized cost.
# %
print('\nFeedforward Using Neural Network ...\n')
# % Weight regularization parameter (we set this to 0 here).
lmbd = 0
J, grad = nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lmbd)
print('Cost at parameters (loaded from ex4weights): %f \n(this value should be about 0.287629)\n' % J)
print('\nProgram paused. Press enter to continue.\n')


print('p4 -----------------------------------------------------')
# %% =============== Part 4: Implement Regularization ===============
# %  Once your cost function implementation is correct, you should now
# %  continue to implement the regularization with the cost.
# %
print('\nChecking Cost Function (w/ Regularization) ... \n')
# % Weight regularization parameter (we set this to 1 here).
lmbd = 1
J, _ = nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lmbd)
print('Cost at parameters (loaded from ex4weights): %f\n(this value should be about 0.383770)\n' % J)
print('Program paused. Press enter to continue.\n')


print('p5 -----------------------------------------------------')
# %% ================ Part 5: Sigmoid Gradient  ================
# %  Before you start implementing the neural network, you will first
# %  implement the gradient for the sigmoid function. You should complete the
# %  code in the sigmoidGradient.m file.
# %
print('\nEvaluating sigmoid gradient...\n')
g = sigmoidGradient(np.array([-1, -0.5, 0, 0.5, 1]))
print('Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:\n  ')
# Format output decimal precisions
np.set_printoptions(formatter={'float': '{:0.4f}\t\t'.format})
print('{} '.format(g))
print('\n\n')
print('Program paused. Press enter to continue.\n')


print('p6 -----------------------------------------------------')
# %% ================ Part 6: Initializing Parameters ================
# %  In this part of the exercise, you will be starting to implement a two
# %  layer neural network that classifies digits. You will start by
# %  implementing a function to initialize the weights of the neural network
# %  (randInitializeWeights.m)
print('\nInitializing Neural Network Parameters ...\n')
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)
# % Unroll parameters
initial_nn_params = np.concatenate((initial_Theta1.reshape(Theta1.size, order='F'),
                                    initial_Theta2.reshape(Theta2.size, order='F')))


print('p7 -----------------------------------------------------')
# %% =============== Part 7: Implement Backpropagation ===============
# %  Once your cost matches up with ours, you should proceed to implement the
# %  backpropagation algorithm for the neural network. You should add to the
# %  code you've written in nnCostFunction.m to return the partial
# %  derivatives of the parameters.
# %
print('\nChecking Backpropagation... \n')
# %  Check gradients by running checkNNGradients
checkNNGradients()
print('\nProgram paused. Press enter to continue.\n')


print('p8 -----------------------------------------------------')
# %% =============== Part 8: Implement Regularization ===============
# %  Once your backpropagation implementation is correct, you should now
# %  continue to implement the regularization with the cost and gradient.
# %
print('\nChecking Backpropagation (w/ Regularization) ... \n')
# %  Check gradients by running checkNNGradients
lmbd = 3
checkNNGradients(lmbd)

# % Also output the costFunction debugging values
debug_J, _ = nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lmbd)
# debug_J, _ = nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lmbd)
print('\n\nCost at (fixed) debugging parameters (w/ lambda = {}): {}'
      '\n(for lambda = 3, this value should be about 0.576051)\n\n'.format(lmbd, debug_J))
print('Program paused. Press enter to continue.\n')


print('p9 -----------------------------------------------------')
# %% =================== Part 8: Training NN ===================
# %  You have now implemented all the code necessary to train a neural
# %  network. To train your neural network, we will now use "fmincg", which
# %  is a function which works similarly to "fminunc". Recall that these
# %  advanced optimizers are able to train our cost functions efficiently as
# %  long as we provide them with the gradient computations.
# %
print('\nTraining Neural Network... \n')
# %  After you have completed the assignment, change the MaxIter to a larger
# %  value to see how more training helps.

# %  You should also try different values of lambda
lmbd = 2

# % Now, costFunction is a function that takes in only one argument (the
# % neural network parameters)
start_time = time.time()
result = opt.minimize(fun=nn_cost_function, x0=initial_nn_params, args=(
    input_layer_size, hidden_layer_size, num_labels, X, y, lmbd), method='CG',
                      jac=True, options={'maxiter': 50, 'disp': True})
print('OK,  -------%s seconds' % (time.time() - start_time))
"""
result.x
result.fun
result.jac
"""
nn_params = result.x

# % Obtain Theta1 and Theta2 back from nn_params
Theta1 = nn_params[:hidden_layer_size * (input_layer_size + 1)].reshape(
    hidden_layer_size, input_layer_size + 1, order='F')
Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):].reshape(
    num_labels, hidden_layer_size + 1, order='F')
print('Program paused. Press enter to continue.\n')


print('p10 -----------------------------------------------------')
# %% ================= Part 9: Visualize Weights =================
# %  You can now "visualize" what the neural network is learning by
# %  displaying the hidden units to see what features they are capturing in
# %  the data.
print('\nVisualizing Neural Network... \n')
displayData(Theta1[:, 1:])
print('Program paused. Press enter to continue.\n')


print('p11 -----------------------------------------------------')
# %% ================= Part 10: Implement Predict =================
# %  After training the neural network, we would like to use it to predict
# %  the labels. You will now implement the "predict" function to use the
# %  neural network to predict the labels of the training set. This lets
# %  you compute the training set accuracy.
pred = predict(Theta1, Theta2, X)
print('Training set accuracy: {}'.format(np.mean(pred == y)*100))
