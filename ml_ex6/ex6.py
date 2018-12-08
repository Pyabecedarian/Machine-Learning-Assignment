"""
%% Machine Learning Online Class
%  Exercise 6 | Support Vector Machines
%
%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the
%  exercise. You will need to complete the following functions:
%
%     gaussianKernel.m
%     dataset3Params.m
%     processEmail.m
%     emailFeatures.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
"""
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn import svm

from ml_ex6.plotData import plotData
from ml_ex6.visualizeBoundaryLinear import visualizeBoundaryLinear
from ml_ex6.gaussionKernel import gaussianKernel
from ml_ex6.visualizeBoundary import visualizeBoundary
from ml_ex6.dataset3Params import dataset3Params

print('p1', '---' * 20)
# %% =============== Part 1: Loading and Visualizing Data ================
# %  We start the exercise by first loading and visualizing the dataset.
# %  The following code will load the dataset into your environment and plot
# %  the data.
# %
print('Loading and Visualizing Data ...')
# % Load from ex6data1:
# % You will have X, y in your environment
Data = sio.loadmat('ex6data1.mat')
X = Data['X']
y = Data['y'].flatten()

# % Plot training data
plotData(X, y)
plt.show()
print('Program paused. Press enter to continue.\n')

print('p2', '---' * 20)
# %% ==================== Part 2: Training Linear SVM ====================
# %  The following code will train a linear SVM on the dataset and plot the
# %  decision boundary learned.
# %
#
# % Load from ex6data1:
# % You will have X, y in your environment
print('Training Linear SVM ...')

# % You should try to change the C value below and see how the decision
# % boundary varies (e.g., try C = 1000)
C = 1
# model = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
clf = svm.SVC(kernel='linear', C=C)
clf.fit(X, y)
visualizeBoundaryLinear(X, y, clf)
print('Program paused. Press enter to continue.\n')

print('p3', '---' * 20)
# %% =============== Part 3: Implementing Gaussian Kernel ===============
# %  You will now implement the Gaussian kernel to use
# %  with the SVM. You should complete the code in gaussianKernel.m
# %
print('\nEvaluating the Gaussian Kernel ...')
x1 = np.array([1, 2, 1])
x2 = np.array([0, 4, -1])
sigma = 2
sim = gaussianKernel(x1, x2, sigma)
print('Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = %f :'
      '\n\t%f\n(for sigma = 2, this value should be about 0.324652)\n' % (sigma, sim))
print('Program paused. Press enter to continue.\n')

print('p4', '---' * 20)
# %% =============== Part 4: Visualizing Dataset 2 ================
# %  The following code will load the next dataset into your environment and
# %  plot the data.
# %
print('Loading and Visualizing Data ...\n')
# % Load from ex6data2:
# % You will have X, y in your environment
Data = sio.loadmat('ex6data2.mat')
X = Data['X']
y = Data['y'].flatten()

# Plot training data
plotData(X, y)
plt.show()
print('Program paused. Press enter to continue.\n')

print('p5', '---' * 20)
# %% ========== Part 5: Training SVM with RBF Kernel (Dataset 2) ==========
# %  After you have implemented the kernel, we can now use it to train the
# %  SVM classifier.
# %
# % SVM Parameters
C = 1
sigma = 0.1
gamma = 1 / (2 * sigma ** 2)

# % We set the tolerance and max_passes lower here so that the code will run
# % faster. However, in practice, you will want to run the training to
# % convergence.
# NOTE:
# The kernel function can be any of the following:
#
# 1.  linear:      <x, x'>
# 2.  polynomial:  (γ<x, x'> + r)^d
#              where d is specified by keyword degree, r by coef0.
# 3.  rbf:         exp(-γ||x - x'||^2)
#              where γ is specified by keyword gamma, must be greater than 0.
# 4.  sigmoid(tanh(γ<x, x'> + r))
#              where r is specified by coef0.
clf = svm.SVC(kernel='rbf', C=C, gamma=gamma)
clf.fit(X, y)
visualizeBoundary(X, y, clf)
print('Program paused. Press enter to continue.\n')

print('p6', '---' * 20)
# %% =============== Part 6: Visualizing Dataset 3 ================
# %  The following code will load the next dataset into your environment and
# %  plot the data.
# %
print('Loading and Visualizing Data ...\n')
# % Load from ex6data3:
# % You will have X, y in your environment
Data = sio.loadmat('ex6data3.mat')
X = Data['X']
y = Data['y'].flatten()
Xval = Data['Xval']
yval = Data['yval'].flatten()
plotData(X, y)
plt.show()
print('Program paused. Press enter to continue.\n')

print('p7', '---' * 20)
# %% ========== Part 7: Training SVM with RBF Kernel (Dataset 3) ==========
#
# %  This is a different dataset that you can use to experiment with. Try
# %  different values of C and sigma here.
# %

# % Try different SVM Parameters here
C, sigma = dataset3Params(X, y, Xval, yval)
# % Train the SVM
clf = svm.SVC(kernel='rbf', C=C, gamma=1/(2*sigma**2))
clf.fit(X, y)
visualizeBoundary(X, y, clf)