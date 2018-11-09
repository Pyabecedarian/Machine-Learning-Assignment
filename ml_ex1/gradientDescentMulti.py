import numpy as np
from ml_ex1.computeCostMulti import computeCostMulti

def gradientDescentMulti(X, y, theta, alpha, num_iters):
    """
    %GRADIENTDESCENTMULTI Performs gradient descent to learn theta
    theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
    taking num_iters gradient steps with learning rate alpha
    :param X_norm:  X.shape = (m, 3)
    :param y:  y.shape = (m, )
    :param theta:
    :param alpha:  Step length of each iteration
    :param num_iters: number of steps to iterate
    :return: theta, J_history
    """

# % Initialize some useful values
    m = y.size   # number of training examples
    J_history = np.zeros(num_iters)

    for i in range(num_iters):

        # % ====================== YOUR CODE HERE ======================
        # % Instructions: Perform a single gradient step on the parameter vector
        # %               theta.
        # %
        # % Hint: While debugging, it can be useful to print out the values
        # %       of the cost function (computeCostMulti) and gradient here.
        # %
        delta = ((theta @ X.T - y) @ X) / m
        theta = theta - alpha * delta

    # % ============================================================

    # % Save the cost J in every iteration
        J_history[i] = computeCostMulti(X, y, theta)

    return theta, J_history


if __name__ == '__main__':
    Data2 = np.loadtxt('ex1data2.txt', delimiter=',')
    X = Data2[:,:2]
    y = Data2[:, 2]
    m = y.size

    from ml_ex1.featureNormalization import featureNormalize
    X, mu, sigma = featureNormalize(X)

    X = np.hstack((np.ones(m).reshape(m, 1), X))
    theta = np.zeros(3)

    theta, J_history = gradientDescentMulti(X, y, theta, 0.3, 20)
    # Turns out alpha = 0.5 is good.  (In fact, alpha can be proximately equal but less then 1.)
    print(theta)
