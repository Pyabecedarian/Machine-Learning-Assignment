import numpy as np


def featureNormalize(X):
    """
    FEATURENORMALIZE Normalizes the features in X
    FEATURENORMALIZE(X) returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation
    is 1. This is often a good preprocessing step to do when
    working with learning algorithms.
    :param X: X.shape = (m, 2)
    :return:  X_norm, mu, sigma
    """
    m = X[:,0].size  # num of training sets


    # You need to set these values correctly
    X_norm = X
    mu = np.zeros(X.ndim)
    sigma = np.zeros(X.ndim)

    #  ====================== YOUR CODE HERE ======================
    #  Instructions: First, for each feature dimension, compute the mean
    #                of the feature and subtract it from the dataset,
    #                storing the mean value in mu. Next, compute the
    #                standard deviation of each feature and divide
    #                each feature by it's standard deviation, storing
    #                the standard deviation in sigma.
    #
    #                Note that X is a matrix where each column is a
    #                feature and each row is an example. You need
    #                to perform the normalization separately for
    #                each feature.
    mu = np.sum(X, axis=0) / m                        # Sum all element in each column
    sigma = np.std(X, axis=0)                         # Standard Deviation in each column
    # sigma = np.max(X,axis=0) - np.min(X, axis=0)    # Max - Min in each column

    # for i in range(X.ndim):                         # 1st
    #     X_norm[:,i] = (X[:,i] - mu[i]) / sigma[i]   # 1st
    X_norm = (X - mu) / sigma

    return X_norm, mu, sigma


if __name__ == '__main__':
    # Data2 = np.loadtxt('ml_ex1/ex1data2.txt', delimiter=',')
    Data2 = np.loadtxt('ex1data2.txt', delimiter=',')
    X = Data2[:,:2]
    y = Data2[:, 2]

    m = y.size

    X_norm, mu, sigma = featureNormalize(X)
