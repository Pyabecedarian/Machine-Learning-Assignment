import numpy as np
from sklearn import svm


def dataset3Params(X, y, Xval, yval):
    """
    DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
    where you select the optimal (C, sigma) learning parameters to use for SVM
    with RBF kernel
      [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and
      sigma. You should complete this function to return the optimal C and
      sigma based on a cross-validation set.
    :param X:
    :param y:
    :param Xval:
    :param yval:
    :return:
    """

    # % You need to return the following variables correctly.
    # C = 1
    # sigma = 0.3

    # % ====================== YOUR CODE HERE ======================
    # % Instructions: Fill in this function to return the optimal C and sigma
    # %               learning parameters found using the cross validation set.
    # %               You can use svmPredict to predict the labels on the cross
    # %               validation set. For example,
    # %                   predictions = svmPredict(model, Xval);
    # %               will return the predictions on the cross validation set.
    # %
    # %  Note: You can compute the prediction error using
    # %        mean(double(predictions ~= yval))
    #
    test_params = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    errs = {}
    for C in test_params:
        for sigma in test_params:
            gamma = 1 / (2 * sigma ** 2)
            clf = svm.SVC(kernel='rbf', C=C, gamma=gamma)
            clf.fit(X, y)
            pred_yval = clf.predict(Xval)
            err = np.mean(pred_yval != yval)
            errs.setdefault(err, []).append((C, sigma))

    C, sigma = errs.get(min(errs))[0]
    return C, sigma
    # % =========================================================================
