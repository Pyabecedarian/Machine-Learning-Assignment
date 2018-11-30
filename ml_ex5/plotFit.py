import numpy as np

from ml_ex5.featureNormalize import featureNormalize
from ml_ex5.polyFeatures import polyFeatures


def polyFit(fit_x, theta, p, miu, sigma):
    fit_x_poly = polyFeatures(fit_x, p)
    fit_x_poly = featureNormalize(fit_x_poly, miu, sigma)
    fit_x_poly = np.c_[np.ones(fit_x.shape[0]), fit_x_poly]
    fit_y = fit_x_poly @ theta

    return fit_y
