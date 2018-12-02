import numpy as np
import scipy.optimize as opt
from ml_ex5.linearRegCostFunction import linearRegCostFunction


def trainLinearReg(X, y, lmbd):
    """
    TRAINLINEARREG Trains linear regression given a dataset (X, y) and a
    regularization parameter lambda
    [theta] = TRAINLINEARREG (X, y, lambda) trains linear regression using
    the dataset (X, y) and regularization parameter lambda. Returns the
    trained parameters theta.

    :param X:  X.shape = (12, 2)
    :param y:  y.shape = (12,  )
    :param lmbd:
    :return: optimal θ
    """
    init_theta = np.ones(X.shape[1])
    # result = opt.minimize(fun=lambda theta: linearRegCostFunction(X, y, theta, lmbd),
    #                       x0=init_theta, jac=True, method='CG',
    #                       options={'maxiter': 400, 'disp': False})
    result = opt.minimize(fun=lambda theta: linearRegCostFunction(X, y, theta, lmbd),
                          x0=init_theta, jac=True, method='CG')
    return result.x
