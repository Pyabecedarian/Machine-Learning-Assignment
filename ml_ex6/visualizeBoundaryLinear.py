import matplotlib.pyplot as plt
import numpy as np

from ml_ex6.plotData import plotData


def visualizeBoundaryLinear(X, y, clf):
    """
    VISUALIZEBOUNDARYLINEAR plots a linear decision boundary learned by the
    SVM
    VISUALIZEBOUNDARYLINEAR(X, y, model) plots a linear decision boundary
    learned by the SVM and overlays the data on it
    """
    xmin = np.min(X[:, 0] - 2)
    xmax = np.max(X[:, 0] + 2)
    ymin = np.min(X[:, 1] - 1)
    ymax = np.max(X[:, 1] + 1)
    xx = np.linspace(xmin, xmax, 50)
    yy = np.linspace(ymin, ymax, 50)
    XX, YY = np.meshgrid(xx, yy)
    xy = np.vstack((XX.ravel(), YY.ravel())).T
    Z = clf.decision_function(xy).reshape(XX.shape)
    plt.contour(XX, YY, Z, levels=[-1, 0, 1], linestyles=['--', '-', '--'])
    plt.scatter(clf.support_vectors_[:,0], clf.support_vectors_[:,1], s=100,
                linewidths=1, facecolor='none', edgecolors='k')
    plotData(X, y)
    plt.show()
