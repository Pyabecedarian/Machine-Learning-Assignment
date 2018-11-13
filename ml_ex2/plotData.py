import numpy as np
import matplotlib.pyplot as plt


def plotData(X, y):
    """
    PLOTDATA Plots the data points X and y into a new figure
    PLOTDATA(x,y) plots the data points with + for the positive examples
    and o for the negative examples. X is assumed to be a Mx2 matrix.
    """

    # % Create New Figure
    fig = plt.figure()

    # % ====================== YOUR CODE HERE ======================
    # % Instructions: Plot the positive and negative examples on a
    # %               2D plot, using the option 'k+' for the positive
    # %               examples and 'ko' for the negative examples.
    # %
    pos = np.nonzero(y == 1)  # np.nonzero() returns a tuple
    neg = np.nonzero(y == 0)  # np.where() instead, returns a tuple

    plt.scatter(X[pos, 0], X[pos, 1], c='black', marker='+')
    plt.scatter(X[neg, 0], X[neg, 1], c='yellow', marker='o', edgecolors='black')
    return fig
    # % =========================================================================
