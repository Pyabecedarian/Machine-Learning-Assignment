import numpy as np
import scipy.optimize as opt
from ml_ex3.lrCostFunction import lrCostFunction
import scipy.io as sio


def oneVsAll(X, y, num_labels, lmbd):
    """
    ONEVSALL trains multiple logistic regression classifiers and returns all
    the classifiers in a matrix all_theta, where the i-th row of all_theta
    corresponds to the classifier for label i
       [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
       logistic regression classifiers and returns each of these classifiers
       in a matrix all_theta, where the i-th row of all_theta corresponds
       to the classifier for label i

    NOTE:
          X.shape = (5000, 400)    y.shape = (5000, )

    About the 'result' ---> opt.OptimizeResult:
        'result' is an opt.OptimizeResult object which contains some useful attributes shown below:
            1).  x         (ndarray) :  Solution of the optimization
            2).  success      (bool) :  Whether of not the optimizer exited successfully
            3).  fun       (ndarray) :  Value of the Cost ------ J
            4).  jav       (ndarray) :  Value of the Gradient -- grad
    """

    # % Some useful variables
    m, n = X.shape
    # % You need to return the following variables correctly
    all_theta = np.zeros((num_labels, n + 1))
    # % Add ones to the X data matrix
    X = np.hstack((np.ones(m).reshape(m, 1), X))

    # % ====================== YOUR CODE HERE ======================
    # % Instructions: You should complete the following code to train num_labels
    # %               logistic regression classifiers with regularization
    # %               parameter lambda.
    # %
    # % Hint: theta(:) will return a column vector.
    # %
    # % Hint: You can use y == c to obtain a vector of 1's and 0's that tell you
    # %       whether the ground truth is true/false for this class.
    # %
    # % Note: For this assignment, we recommend using fmincg to optimize the cost
    # %       function. It is okay to use a for-loop (for c = 1:num_labels) to
    # %       loop over the different classes.
    # %
    # %       fmincg works similarly to fminunc, but is more efficient when we
    # %       are dealing with large number of parameters.
    # %
    for i in range(1, num_labels+1):  # If i=10, this is a classifier of "0"
        yi = 1 * (y == i)  # If i=10, the digit classifier is of "0's"
        init_theta = np.zeros(n + 1)
        result = opt.minimize(fun=lrCostFunction, x0=init_theta, args=(X, yi, lmbd),
                              method='CG', jac=True)
        all_theta[i-1, :] = result.x

    return all_theta
    # % =========================================================================


if __name__ == '__main__':
    data_dict = sio.loadmat('ex3data1.mat')  # % training data stored in arrays X, y
    X = data_dict['X']
    y = data_dict['y'].flatten()
    m, n = X.shape
    yi = 1 * (y == 1)
    lmbd = 0.1
    init_theta = np.zeros(n + 1)
    # result = opt.minimize(fun=lrCostFunction, x0=init_theta, args=(X, yi, lmbd),
    #                       method='CG', jac=True)

    all_theta = oneVsAll(X, y, 10, lmbd=0.1)
    print('X.shape--1', X.shape)
    print('all_theta.shape--1', all_theta.shape)
