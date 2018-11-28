import numpy as np


def randInitializeWeights(L_in, L_out):
    """
    RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
    incoming connections and L_out outgoing connections
    W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights
    of a layer with L_in incoming connections and L_out outgoing
    connections.

    Note that W should be set to a matrix of size(L_out, 1 + L_in) as
    the first column of W handles the "bias" terms
    """
    # % You need to return the following variables correctly
    # W = np.zeros((L_out, 1 + L_in))

    # % ====================== YOUR CODE HERE ======================
    # % Instructions: Initialize W randomly so that we break the symmetry while
    # %               training the neural network.
    # %
    # % Note: The first column of W corresponds to the parameters for the bias unit
    # %
    epsilon_init = np.around(np.sqrt(6) / np.sqrt(L_in+L_out), decimals=2)  # 0.12
    theta_init = np.random.rand(L_out, 1+L_in)*2*epsilon_init - epsilon_init
    return theta_init


if __name__ == '__main__':
    e = randInitializeWeights(4, 4)
    print(abs(e)<=0.12)
