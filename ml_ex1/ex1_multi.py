"""
%% Machine Learning Online Class
%  Exercise 1: Linear regression with multiple variables
%
%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the
%  linear regression exercise.
%
%  You will need to complete the following functions in this
%  exericse:
%
%     warmUpExercise.m
%     plotData.m
%     gradientDescent.m
%     computeCost.m
%     gradientDescentMulti.m
%     computeCostMulti.m
%     featureNormalize.m
%     normalEqn.m
%
%  For this part of the exercise, you will need to change some
%  parts of the code below for various experiments (e.g., changing
%  learning rates).
"""
import numpy as np
import matplotlib.pyplot as plt

# ================ Part 1: Feature Normalization ================
# Clear andfprintf('Loading data ...\n');
print('Loading data ...\n')

# Load Data
Data2 = np.loadtxt('ex1data2.txt', delimiter=',')
# Data2 = np.loadtxt('ml_ex1/ex1data2.txt', delimiter=',')
X = Data2[:,:2]
y = Data2[:,2]

m = y.size  # num of training set

# Print out some data points
print('First 10 examples from the dataset: \n')
X_10 = zip(X[:10,0], X[:10,1], y[:10])
for x0_i, x1_i, y_i in X_10:
    print('\t x = [ {}, {} ],\ty = {}'.format(x0_i, x1_i, y_i))

# input('Program paused. Press enter to continue.\n')

# Scale features and set them to zero mean
print('Normalizing Features ...\n')

from ml_ex1.featureNormalization import featureNormalize
X_norm, mu, sigma = featureNormalize(X)

# % Add intercept term to X
X = np.hstack((np.ones(m).reshape(m,1), X))
X_norm = np.hstack((np.ones(m).reshape(m,1), X_norm))


# %% ================ Part 2: Gradient Descent ================
#
# % ====================== YOUR CODE HERE ======================
# % Instructions: We have provided you with the following starter
# %               code that runs gradient descent with a particular
# %               learning rate (alpha).
# %
# %               Your task is to first make sure that your functions -
# %               computeCost and gradientDescent already work with
# %               this starter code and support multiple variables.
# %
# %               After that, try running gradient descent with
# %               different values of alpha and see which one gives
# %               you the best result.
# %
# %               Finally, you should complete the code at the end
# %               to predict the price of a 1650 sq-ft, 3 br house.
# %
# % Hint: By using the 'hold on' command, you can plot multiple
# %       graphs on the same figure.
# %
# % Hint: At prediction, make sure you do the same feature normalization.
# %
print('Running gradient descent ...\n')
alpha = 0.3
num_iters = 50
# % Init Theta and Run Gradient Descent
theta = np.zeros(3)


# Implement gradient descent of multi-variables in cost function J
from ml_ex1.gradientDescentMulti import gradientDescentMulti
theta, J_history = gradientDescentMulti(X_norm, y, theta, alpha, num_iters)

# Plot the convergence graph
plt.figure(1)
plt.plot(np.arange(1,num_iters+1), J_history, c='b', linewidth=1, label='Convergence Line')
plt.xlabel('Num of  iterations')
plt.ylabel('Cost J')
plt.title('Convergence of Gradient Descent')
plt.show()

# % Display gradient descent's result
print('Theta computed from gradient descent: \n')
print(' {} \n'.format(theta))
print('\n')

# % Estimate the price of a 1650 sq-ft, 3 br house
# % ====================== YOUR CODE HERE ======================
# % Recall that the first column of X is all-ones. Thus, it does
# % not need to be normalized.
x = np.array([1650, 3])
# ****IMPORTANT***: DON'T FORGET TO NORMALIZE THE FEATURES WHEN MAKE PREDICTIONS!!!!
x = (x - mu) / sigma
x = np.hstack((np.ones(1), x))
price = theta @ x.T  # % You should change this

# % ============================================================
print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n $%f \n' % price)  # 293092.212731
# input('Program paused. Press enter to continue.\n')

# %% ================ Part 3: Normal Equations ================
print('Solving with normal equations...\n')
#
# % ====================== YOUR CODE HERE ======================
# % Instructions: The following code computes the closed form
# %               solution for linear regression using the normal
# %               equations. You should complete the code in
# %               normalEqn.m
# %
# %               After doing so, you should complete this code
# %               to predict the price of a 1650 sq-ft, 3 br house.
# %

# %% Load Data
Data3 = np.loadtxt('ex1data2.txt', delimiter=',')
X_ = Data3[:,:2]
y_ = Data3[:, 2]

# % Add intercept term to
X_ = np.hstack((np.ones(m).reshape(m,1), X_))

# % Calculate the parameters from the normal equation
from ml_ex1.normalEqn import normalEqn
theta = normalEqn(X_, y_)

# % Display normal equation's result
print('Theta computed from the normal equations: \n')
print(' {} \n'.format(theta))
print('\n')

# % Estimate the price of a 1650 sq-ft, 3 br house
# % ====================== YOUR CODE HERE ======================
x = np.array([1, 1650, 3])
price = theta @ x  # % You should change this
#
#
# % ============================================================

print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations):\n $%f\n' % price)

