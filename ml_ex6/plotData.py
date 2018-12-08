import numpy as np
import matplotlib.pyplot as plt


def plotData(X, y):
    """
    PLOTDATA Plots the data points X and y into a new figure
    PLOTDATA(x,y) plots the data points with + for the positive examples
    and o for the negative examples. X is assumed to be a Mx2 matrix.

    Note: This was slightly modified such that it expects y = 1 or y = 0
    :param X:
    :param y:
    :return:
    """
    # % Find Indices of Positive and Negative Examples
    pos = X[y == 1]
    neg = X[y == 0]

    # Plot example
    plt.scatter(pos[:, 0], pos[:, 1], c='black', edgecolors=None, marker='+')
    plt.scatter(neg[:, 0], neg[:, 1], c='yellow', edgecolors='black', marker='o')
    plt.xlim(np.min(X) - 0.1, np.max(X) + 0.1)
