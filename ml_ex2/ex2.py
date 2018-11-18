"""
%% Machine Learning Online Class - Exercise 2: Logistic Regression
%
%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the logistic
%  regression exercise. You will need to complete the following functions
%  in this exericse:
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

from ml_ex2.costFunction import costFunction
from ml_ex2.plotData import plotData
from ml_ex2.plotDecisionBoundary import plotDecisionBoundary
from ml_ex2.predict import predict
from ml_ex2.sigmoid import sigmoid

# %% Load Data
# %  The first two columns contains the exam scores and the third column
# %  contains the label.
data = np.loadtxt('./ex2data1.txt', delimiter=',')
X = data[:,:2]
y = data[:, 2]

# %% ==================== Part 1: Plotting ====================
# %  We start the exercise by first plotting the data to understand the
# %  the problem we are working with.

print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.\n')
plotData(X, y)
plt.legend(['Admitted', 'Not admitted'], loc='upper right')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.show()


# input('\nProgram paused. Press enter to continue.\n')


# %% ============ Part 2: Compute Cost and Gradient ============
# %  In this part of the exercise, you will implement the cost and gradient
# %  for logistic regression. You need to complete the code in
# %  costFunction.m
#
# %  Setup the data matrix appropriately, and add ones for the intercept term
m, n = X.shape
X = np.hstack((np.ones(m).reshape(m,1), X))  # Use X = np.c_[np.ones(m), X] instead
initial_theta = np.zeros(n+1)

# % Compute and display initial cost and gradient
cost, grad = costFunction(initial_theta, X, y)

# np.set_printoptions(formatter={'float': '{:0.4f}\n'.format})

print('Cost at initial theta (zeros): {}\n'.format(cost))
print('Expected cost (approx): 0.693\n')
print('Gradient at initial theta (zeros): \n')
print(' {} \n'.format(grad))
print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n')

# % Compute and display cost and gradient with non-zero theta
test_theta = np.array([-24, 0.2, 0.2])
cost, grad = costFunction(test_theta, X, y)

print('\nCost at test theta: {}\n'.format(cost))
print('Expected cost (approx): 0.218\n')
print('Gradient at test theta: \n')
print(' {} \n'.format(grad))
print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n')


# input('\nProgram paused. Press enter to continue.\n')


# %% ============= Part 3: Optimizing using fminunc  =============
# %  In this exercise, you will use a built-in function (fminunc) to find the
# %  optimal parameters theta.

# %  ☆☆☆   "Newton Conjugate-Gradient"
# %  NOTE: "func" should return both J and ∂J/∂θ
theta, *_ = opt.fmin_tnc(func=costFunction, x0=initial_theta, args=(X, y))  # fmin_tnc() returns a tuple
# theta = result[0]  # Optimal theta
cost = costFunction(theta, X, y)[0]  # min cost

# %  Print theta to screen
print('Cost at theta found by fmin_tnc: {:0.4f}\n'.format(cost))
print('Expected cost (approx): 0.203\n')
print('theta: \n')
print(' {} \n'.format(theta))
print('Expected theta (approx):\n')
print(' [-25.161    0.206    0.201]\n')

# % Plot Boundary
plotDecisionBoundary(theta, X, y)
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')

# input('\nProgram paused. Press enter to continue.\n')


# %% ============== Part 4: Predict and Accuracies ==============
# %  After learning the parameters, you'll like to use it to predict the outcomes
# %  on unseen data. In this part, you will use the logistic regression model
# %  to predict the probability that a student with score 45 on exam 1 and
# %  score 85 on exam 2 will be admitted.
# %
# %  Furthermore, you will compute the training and test set accuracies of
# %  our model.
# %
# %  Your task is to complete the code in predict.m
#
# %  Predict probability for a student with score 45 on exam 1
# %  and score 85 on exam 2

prob = sigmoid(np.array([1,45,85]) @ theta)
print('For a student with scores 45 and 85, we predict an admission probability of %f\n' % prob)
print('Expected value: 0.775 ± 0.002\n\n')

# % Compute accuracy on our training set
p = predict(theta, X)
a = 100 * np.mean(1 * (p==y))
print('Train Accuracy:{}'.format(a))
print('Expected accuracy (approx): 89.0\n')
print('\n')
