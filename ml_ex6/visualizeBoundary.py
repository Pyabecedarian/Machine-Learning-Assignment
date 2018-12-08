import numpy as np
import matplotlib.pyplot as plt
from ml_ex6.plotData import plotData


def visualizeBoundary(X, y, clf):
    """
    VISUALIZEBOUNDARY plots a non-linear decision boundary learned by the SVM
    VISUALIZEBOUNDARY(X, y, model) plots a non-linear decision
    boundary learned by the SVM and overlays the data on it
    """
    xx = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
    yy = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 100)
    XX, YY = np.meshgrid(xx, yy)
    xy = np.vstack((XX.ravel(), YY.ravel())).T
    Z = clf.decision_function(xy).reshape(XX.shape)
    plt.contour(XX, YY, Z, levels=0)
    plotData(X, y)
    plt.show()
