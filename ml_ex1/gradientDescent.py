import numpy as np
from ml_ex1.computeCost import computeCost

def gradientDescent(X, y, theta, alpha, num_iters):
    """
    GRADIENTDESCENT Performs gradient descent to learn theta
    theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by
    taking num_iters gradient steps with learning rate alpha
    """

    # Initialize some useful values
    m = y.size
    J_history = np.zeros(num_iters)

    for i in range(num_iters):
        delta = ((theta@X.T - y) @ X) / m

        # Update theta in every iteration
        theta = theta - alpha * delta

        # Save the cost J in every iteration
        J_history[i] = computeCost(X, y, theta)

    return theta, J_history
