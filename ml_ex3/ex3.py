"""
%% Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all

%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the
%  linear exercise. You will need to complete the following functions
%  in this exericse:
%
%     lrCostFunction.m (logistic regression cost function)
%     oneVsAll.m
%     predictOneVsAll.m
%     predict.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%
"""
import scipy.io as sio
import numpy as np
from ml_ex3.displayData import displayData
from ml_ex3.lrCostFunction import lrCostFunction
from ml_ex3.oneVsAll import oneVsAll
from ml_ex3.predictOneVsAll import predictOneVsAll

# %% Setup the parameters you will use for this part of the exercise
input_layer_size = 400  # % 20x20 Input Images of Digits
num_labels = 10  # % 10 labels, from 1 to 10 (note that we have mapped "0" to label 10).

# %% =========== Part 1: Loading and Visualizing Data =============
# %  We start the exercise by first loading and visualizing the dataset.
# %  You will be working with a dataset that contains handwritten digits.
# %

# % Load Training Data
print('Loading and Visualizing Data ...\n')

data_dict = sio.loadmat('ex3data1.mat')  # % training data will be stored in arrays X, y
X = data_dict['X']
y = data_dict['y'].flatten()

m = X[:, 1].size

# % Randomly select 100 data points to display
rand_indices = np.random.permutation(m)
sel = X[rand_indices[:100], :]

displayData(sel)

# input('Program paused. Press enter to continue.\n')


# %% ============ Part 2a: Vectorize Logistic Regression ============
# %  In this part of the exercise, you will reuse your logistic regression
# %  code from the last exercise. You task here is to make sure that your
# %  regularized logistic regression implementation is vectorized. After
# %  that, you will implement one-vs-all classification for the handwritten
# %  digit dataset.
# %

# % Test case for lrCostFunction
print('\nTesting lrCostFunction() with regularization')
#
theta_t = np.array([-2, -1, 1, 2])
X_t = np.hstack((np.ones(5).reshape(5, 1), np.arange(1, 16).reshape(3, 5).T / 10))
y_t = 1 * (np.array([1, 0, 1, 0, 1]) >= 0.5)
lambda_t = 3
# J, grad = lrCostFunction(theta_t, X_t, y_t, lambda_t)
J, grad = lrCostFunction(theta_t, X_t, y_t, lambda_t)
print('\nCost: %f\n' % J)
print('Expected cost: 2.534819\n')
print('Gradients:\n')
print(' {} \n'.format(grad))
print('Expected gradients:\n')
print(' 0.146561    -0.548558    0.724722   1.398003\n')

# input('Program paused. Press enter to continue.\n')


# %% ============ Part 2b: One-vs-All Training ============
print('\nTraining One-vs-All Logistic Regression...\n')

lmbd = 0.1
all_theta = oneVsAll(X, y, num_labels, lmbd)

# input('Program paused. Press enter to continue.\n')


# %% ================ Part 3: Predict for One-Vs-All ================
pred = predictOneVsAll(all_theta, X)
print('\nTraining Set Accuracy: %f\n' % (np.mean((pred == y) * 1.0)))
