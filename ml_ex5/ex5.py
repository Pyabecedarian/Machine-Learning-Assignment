"""
%% Machine Learning Online Class
%  Exercise 5 | Regularized Linear Regression and Bias-Variance
%
%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the
%  exercise. You will need to complete the following functions:
%
%     linearRegCostFunction.m
%     learningCurve.m
%     validationCurve.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%
"""
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from ml_ex5.featureNormalize import featureNormalize
from ml_ex5.learningCurve import learningCurve
from ml_ex5.linearRegCostFunction import linearRegCostFunction
from ml_ex5.plotFit import polyFit
from ml_ex5.polyFeatures import polyFeatures
from ml_ex5.trainLinearReg import trainLinearReg
from ml_ex5.validationCurve import validationCurve

print('p1', '--'*20)
# %% =========== Part 1: Loading and Visualizing Data =============
# %  We start the exercise by first loading and visualizing the dataset.
# %  The following code will load the dataset into your environment and plot
# %  the data.
# %

# % Load Training Data
print('Loading and Visualizing Data ...\n')
# % Load from ex5data1:
# % You will have X, y, Xval, yval, Xtest, ytest in your environment
Data = sio.loadmat('ex5data1.mat')
X = Data['X'].flatten()                    #     X.shape = (12, 1)
y = Data['y'].flatten()                    #     y.shape = (12, 1)
Xval = Data['Xval'].flatten()              #  Xval.shape = (21, 1)
yval = Data['yval'].flatten()              #  yval.shape = (21, 1)
Xtest = Data['Xtest'].flatten()            # Xtest.shape = (21, 1)
ytest = Data['ytest'].flatten()            # ytest.shape = (21, 1)

# %  Plot training data
plt.figure()
plt.scatter(X, y, c='r', marker='x', label='Training Data')
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
print('Program paused. Press enter to continue.\n')


print('p2', '--'*20)
# %% =========== Part 2: Regularized Linear Regression Cost =============
# %  You should now implement the cost function for regularized linear
# %  regression.
# %
theta = np.array([1, 1])
J, _ = linearRegCostFunction(np.c_[np.ones(X.shape[0]), X], y, theta, 1)
print('Cost at theta = [1 ; 1]: %f\n(this value should be about 303.993192)\n' % J)
print('Program paused. Press enter to continue.\n')


print('p3', '--'*20)
# %% =========== Part 3: Regularized Linear Regression Gradient =============
# %  You should now implement the gradient for regularized linear
# %  regression.
# %
theta = np.array([1, 1])
_, grad = linearRegCostFunction(np.c_[np.ones(X.shape[0]), X], y, theta, 1)
print('Gradient at theta = [1 ; 1]:  [{:.6f}; {:.6f}]\n'
      '(this value should be about [-15.303016; 598.250744])\n'.format(grad[0], grad[1]))
print('Program paused. Press enter to continue.\n')


print('p4', '--'*20)
# %% =========== Part 4: Train Linear Regression =============
# %  Once you have implemented the cost and gradient correctly, the
# %  trainLinearReg function will use your cost function to train
# %  regularized linear regression.
# %
# %  Write Up Note: The data is non-linear, so this will not give a great
# %                 fit.
# %

# %  Train linear regression with lambda = 0
lmbd = 0
theta = trainLinearReg(np.c_[np.ones(X.shape[0]), X], y, lmbd)
plt.plot(X, np.c_[np.ones(X.shape[0]), X]@theta, c='blue', linewidth=3, label='Linear Fit')
plt.legend(loc='upper left')
plt.show()
print('Program paused. Press enter to continue.\n')


print('p5', '--'*20)
# %% =========== Part 5: Learning Curve for Linear Regression =============
# %  Next, you should implement the learningCurve function.
# %
# %  Write Up Note: Since the model is underfitting the data, we expect to
# %                 see a graph with "high bias" -- Figure 3 in ex5.pdf
# %
lmbd = 0
error_train, error_val = learningCurve(np.c_[np.ones(X.shape[0]), X], y,
                                       np.c_[np.ones(Xval.shape[0]), Xval], yval, lmbd)
plt.figure()
plt.plot(np.arange(1, X.shape[0]+1), error_train, c='b', linewidth=1, label='Train')     # Training Error
plt.plot(np.arange(1, X.shape[0]+1), error_val, c='g', linewidth=1, label='Cross Validation')  # CV Error
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.xlim(0, 13)
plt.ylim(0, 150)
plt.legend(loc='upper right')
plt.show()

print('# Training Examples\tTrain Error\tCross Validation Error\n')
for i in range(X.shape[0]):
    print('\t{}\t{}\t{}'.format(i+1, error_train[i], error_val[i]))
print('Program paused. Press enter to continue.\n')


print('p6', '--'*20)
# %% =========== Part 6: Feature Mapping for Polynomial Regression =============
# %  One solution to this is to use polynomial regression. You should now
# %  complete polyFeatures to map each example into its powers
# %
p = 8
# % Map X onto Polynomial Features and Normalize
X_poly = polyFeatures(X, p)
X_poly, miu, sigma = featureNormalize(X_poly)
X_poly = np.c_[np.ones(X_poly.shape[0]), X_poly]

# % Map X_poly_val and normalize (using mu and sigma)
Xval_poly = polyFeatures(Xval, p)
Xval_poly = featureNormalize(Xval_poly, miu, sigma)
Xval_poly = np.c_[np.ones(Xval_poly.shape[0]), Xval_poly]

# % Map X_poly_test and normalize (using mu and sigma)
Xtest_poly = polyFeatures(Xtest, p)
Xtest_poly = featureNormalize(Xtest_poly, miu, sigma)
Xtest_poly = np.c_[np.ones(Xtest_poly.shape[0]), Xtest_poly]

print('Normalized Training Example 1:\n')
print('{}\n'.format(X_poly[1, :]))
print('\nProgram paused. Press enter to continue.\n')


print('p7', '--'*20)
# %% =========== Part 7: Learning Curve for Polynomial Regression =============
# %  Now, you will get to experiment with polynomial regression with multiple
# %  values of lambda. The code below runs polynomial regression with
# %  lambda = 0. You should try running the code with different values of
# %  lambda to see how the fit and learning curve change.
# %
lmbd = 3
theta = trainLinearReg(X_poly, y, lmbd)
plt.figure(1)
plt.scatter(X, y, c='r', marker='x', label='Training Data')
fit_x = np.arange(np.min(X)-30, np.max(X)+20, 0.05)
fit_y = polyFit(fit_x, theta, p, miu, sigma)  # NOTE: The fitting curve may not looks like what it shows in the ex5.pdf
plt.plot(fit_x, fit_y)
plt.xlabel('Change in water level (X)')
plt.ylabel('Water flowing out of the dam(y)')
plt.title('Polynomial Regression Fit(λ = %.4f)' % lmbd)
plt.show()


plt.figure(2)
error_train, error_val = learningCurve(X_poly, y, Xval_poly, yval, lmbd)
plt.plot(np.arange(1, X_poly.shape[0]+1), error_train, c='b', linewidth=1, label='Train')     # Training Error
plt.plot(np.arange(1, X_poly.shape[0]+1), error_val, c='g', linewidth=1, label='Cross Validation')  # CV Error
plt.legend(loc='upper right')
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.title('Polynomial Regression Learning Curve (λ = %.4f)' % lmbd)
plt.show()

print('# Training Examples\tTrain Error\tCross Validation Error\n')
for i in range(X.shape[0]):
    print('\t{}\t{}\t{}'.format(i+1, error_train[i], error_val[i]))
print('Program paused. Press enter to continue.\n')


print('p8', '--'*20)
# %% =========== Part 8: Validation for Selecting Lambda =============
# %  You will now implement validationCurve to test various values of
# %  lambda on a validation set. You will then use this to select the
# %  "best" lambda value.
# %
lambda_vec, error_train, error_val = validationCurve(X_poly, y, Xval_poly, yval)
plt.figure()
plt.plot(lambda_vec, error_train, c='b', linewidth=2, label='Train')
plt.plot(lambda_vec, error_val, c='g', linewidth=2, label='Cross Validation')
plt.xlabel('λ')
plt.ylabel('Error')
plt.legend(loc='upper right')
plt.show()

print('lambda\t\tTrain Error\tValidation Error\n')
for i in range(lambda_vec.size):
    print('{:.4f}\t\t{:.4f}\t\t{:.4f}'.format(lambda_vec[i], error_train[i], error_val[i]))


print('p9', '--'*20)
# ================== Part 9: Compute test set error ====================
theta = trainLinearReg(X_poly, y, lmbd=3)
error_test = linearRegCostFunction(Xtest_poly, ytest, theta, lmbd=0)[0]
print('Test set error is %f' % error_test)
