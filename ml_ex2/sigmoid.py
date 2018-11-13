from numpy import exp


def sigmoid(z):
    """
                 1
    g(z) =  -----------
             1 + e^(-z)
    """
    g = 1 / (1 + exp(-z))

    return g
