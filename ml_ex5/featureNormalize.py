import numpy as np


def featureNormalize(X, miu=None, sigma=None):
    """
    FEATURENORMALIZE Normalizes the features in X
    FEATURENORMALIZE(X) returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation
    is 1. This is often a good preprocessing step to do when
    working with learning algorithms.
    :param X: X.shape = (m, p)
    :return:  X_normalized, mu, sigma
    """
    m, p = X.shape
    if not isinstance(miu, None.__class__):
        return (X - miu) / sigma
    miu = np.mean(X, axis=0)  # vector of mean of each column of X
    sigma = np.std(X, axis=0)  # vector of standard deviation of each column of X
    return (X - miu) / sigma, miu, sigma
