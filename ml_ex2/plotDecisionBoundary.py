import numpy as np
import matplotlib.pyplot as plt

from ml_ex2.mapFeature import mapFeature
from ml_ex2.plotData import plotData


def plotDecisionBoundary(theta, X, y):
    """
    function plotDecisionBoundary(theta, X, y)
    PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
    the decision boundary defined by theta
    PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the
    positive examples and o for the negative examples. X is assumed to be
    a either
        1) Mx3 matrix, where the first column is an all-ones column for the
           intercept.
        2) MxN, N>3 matrix, where the first column is all-ones
    """
    # Plot Data
    fig = plotData(X[:,1:], y)

    # Plot Decision Boundary
    if X.shape == (y.size, 3):
        # Decision Boundary is a straight line, where only 2 points
        # is needed to define a line, so choose two endpoints
        plot_x = np.array([min(X[:,1]), max(X[:,1])-8])
        plot_y = (theta[0]*np.ones(2) + theta[1]*plot_x) / (-theta[2])
        plt.plot(plot_x, plot_y, c='blue', linewidth=2, label='Decision Boundary')
    else:
        # Decision Boundary is a contour line, lvl = theta @ X = 0
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        # Evaluate z = X @ theta over the grid
        z = np.zeros((u.size, v.size))
        for i in range(u.size):
            for j in range(v.size):
                z[i, j] = mapFeature(u[i], v[j]) @ theta
        z = z.T  # % important to transpose z before calling contour
        # plt.contour
        # % Plot z = 0
        # % Notice you need to specify the range [0, 0]
        cs = plt.contour(u, v, z, 0)
        plt.legend([cs.collections[0]], ['Decision Boundary'])
    fig.show()
