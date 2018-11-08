import numpy as np


def gradientDescent(X, y, theta, alpha, num_iters):
    """
    GRADIENTDESCENT Performs gradient descent to learn theta
    theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by
    taking num_iters gradient steps with learning rate alpha
    """
    from ml_ex1.computeCost import computeCost

    # Initialize some useful values
    m = y.size
    J_history = np.zeros(num_iters)

    # delta = np.zeros(2)                                                # 1st + 2nd
    for i in range(num_iters):
        # for j in range(2):
        #     # delta[j] = np.sum((theta @ X.T - y) * X[:, j]) / m       # 1st
        #     delta[j] = ((theta @ X.T - y) @ X[:, j]) / m               # 2nd
        # theta = theta - alpha * delta                                  # 1st + 2nd
        delta = ((theta@X.T - y) @ X) / m                                # last

        # Update theta in every iteration
        theta = theta - alpha * delta

        # Save the cost J in every iteration
        J_history[i] = computeCost(X, y, theta)

    return theta, J_history
