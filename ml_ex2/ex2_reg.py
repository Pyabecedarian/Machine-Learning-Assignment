"""
%% Machine Learning Online Class - Exercise 2: Logistic Regression
%
%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the second part
%  of the exercise which covers regularization with logistic regression.
%
%  You will need to complete the following functions in this exericse:
%
%     sigmoid.m
%     costFunction.m
%     predict.m
%     costFunctionReg.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from ml_ex2.costFunctionReg import costFunctionReg
from ml_ex2.plotData import plotData
from ml_ex2.mapFeature import mapFeature
from ml_ex2.plotDecisionBoundary import plotDecisionBoundary

# %% Load Data
# %  The first two columns contains the X values and the third column
# %  contains the label (y).

data = np.loadtxt('./ex2data2.txt', delimiter=',')
X = data[:,:2]
y = data[:, 2]

# plt.figure(figsize=(5,4))
plt.figure()
# plotData(X, y, xlabel='Mircochip Test 1', ylabel='Mircochip Test 2')
plotData(X, y)
plt.xlim(-1, 1.5)
plt.ylim(-0.8, 1.2)
plt.legend(['Admitted', 'Not admitted'], loc='upper right')
plt.show()

# %% =========== Part 1: Regularized Logistic Regression ============
# %  In this part, you are given a dataset with data points that are not
# %  linearly separable. However, you would still like to use logistic
# %  regression to classify the data points.
# %
# %  To do so, you introduce more features to use -- in particular, you add
# %  polynomial features to our data matrix (similar to polynomial
# %  regression).
# %

# Add Polynomial Features.
# Notice that mapFeature also adds a column of 1's
X = mapFeature(X[:,0], X[:,1])
m, n = X.shape
# Initializing theta
init_theta = np.zeros(n)

# Set regularization parameter λ to 1
lambd = 1
# Compute and display initial cost and gradient
cost, grad = costFunctionReg(init_theta, X, y, lambd)

print('Cost at initial theta (zeros): %.3f' % cost)
print('Expected cost (approx): 0.693\n')
print('Gradient at initial theta (zeros) - first five values only:\n')
# Set print options which determine the way floating point numbers, arrays
# and other NumPy objects are displayed.
# np.set_printoptions(suppress=True, precision=4)
print(' {} \n'.format(grad[:5]))
print('Expected gradients (approx) - first five values only:\n')
print(' 0.0085   0.0188   0.0001   0.0503   0.0115')

# %% ============= Part 2: Regularization and Accuracies =============
# %  Optional Exercise:
# %  In this part, you will get to try different values of lambda and
# %  see how regularization affects the decision coundart
# %
# %  Try the following values of lambda (0, 1, 10, 100).
# %
# %  How does the decision boundary change when you vary lambda? How does
# %  the training set accuracy vary?
# %

# % Initialize fitting parameters
m, n = X.shape
init_theta = np.zeros(n)
# init_theta = zeros(size(X, 2), 1)
# % Set regularization parameter lambda to 1 (you should vary this)
# lambd = 0    # Overfitting
lambd = 1    # Just good
# lambd = 100  # Underfitting

# ☆☆☆ Use "Newton Conjugate-Gradient" to optimize the costFunctionReg(theta, X, y)
result = opt.minimize(fun=costFunctionReg, x0=init_theta, args=(X, y, lambd), method='TNC', jac=True)
plotDecisionBoundary(result.x, X, y)
