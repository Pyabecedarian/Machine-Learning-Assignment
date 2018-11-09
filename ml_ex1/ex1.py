import numpy as np
import matplotlib.pyplot as plt

# ==================== Part 1: Basic Function ====================
print('Running warmUpExercise ... \n')
print('5x5 Identity Matrix: \n')

from ml_ex1.warmUpExercise import warmUpExercise
warmUpExercise()

# input('Program paused. Press enter to continue.\n')

# ======================= Part 2: Plotting =======================
print('Plotting Data ...\n')
Data1 = np.loadtxt('ex1data1.txt', delimiter=',')  # use this one when execute the code
# Data1 = np.loadtxt('ml_ex1/ex1data1.txt', delimiter=',')  # use this one when run in console
X = Data1[:,0]
y = Data1[:,1]
m = y.size  # number of training examples

from ml_ex1.plotData import plotData
plt.figure(1)
plotData(X, y)

# input('Program paused. Press enter to continue.\n')

# =================== Part 3: Cost and Gradient descent ===================
X.shape = (m,1)  # Shape X explicitly a Column Vector
X = np.hstack((np.ones(m).reshape(m,1), X))
theta = np.zeros(2)  # initialize fitting parameters

# Some gradient descent settings
iterations = 1500
alpha = 0.01

print('\nTesting the cost function ...\n')
# compute and display initial cost
from ml_ex1.computeCost import computeCost
from ml_ex1.computeCostMulti import computeCostMulti

J = computeCost(X, y, theta)
print('With theta = [0 ; 0]\nCost computed = %f\n' % J)
print('Expected cost value (approx) 32.07\n')

# further testing of the cost function
J = computeCost(X, y, np.array([-1,2]))
print('\nWith theta = [-1 ; 2]\nCost computed = %f\n' % J)
print('Expected cost value (approx) 54.24\n')

# input('Program paused. Press enter to continue.\n')

print('\nRunning Gradient Descent ...\n')
# run gradient descent
from ml_ex1.gradientDescent import gradientDescent
theta, J = gradientDescent(X, y, theta, alpha, iterations)

# print theta to screen
print('Theta found by gradient descent:\n')
print('{}\n'.format(theta))
print('Expected theta values (approx)\n')
print(' -3.6303\n  1.1664\n\n')

# Plot the linear fit
plt.plot(X[:,1], theta@X.T, c='blue', linewidth=3, label='Linear regression')
plt.legend(loc='lower right')
plt.show()

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.array([1, 3.5]) @ theta
print('For population = 35,000, we predict a profit of %f\n' % (predict1*10000))
predict2 = np.array([1, 7]) @ theta
print('For population = 70,000, we predict a profit of %f\n' % (predict2*10000))

# input('Program paused. Press enter to continue.\n')

# ============= Part 4: Visualizing J(theta_0, theta_1) =============
print('Visualizing J(theta_0, theta_1) ...\n')
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

# initialize J_vals to a matrix of 0's
J_vals = np.zeros((theta0_vals.size, theta1_vals.size))

# Fill out J_vals
for i in range(1, theta0_vals.size):
    for j in range(1, theta1_vals.size):
        t = np.array([theta0_vals[i], theta1_vals[j]])
        J_vals[i][j] = computeCost(X, y, t)

# Surface plot
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

fig = plt.figure()
# fig, ax = plt.subplots(2,1)
ax1 = fig.gca(projection='3d')

theta0, theta1 = np.meshgrid(theta0_vals, theta1_vals)
surf = ax1.plot_surface(theta1, theta0, J_vals, cmap=cm.coolwarm,
                       linewidth=0, antialiased=True)
plt.show()

# Contour plot
fig, ax = plt.subplots()
CS = ax.contour(theta0, theta1, J_vals, np.logspace(-2, 3, 20))
ax.clabel(CS, inline=1, fontsize=10)
plt.show()

