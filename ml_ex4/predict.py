import numpy as np

from ml_ex4.sigmoid import sigmoid


def predict(Theta1, Theta2, X):
    """
    PREDICT Predict the label of an input given a trained neural network
    p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
    trained weights of a neural network (Theta1, Theta2)
    """
    m, _ = X.shape

    # % You need to return the following variables correctly
    p = np.zeros(m)

    X = np.c_[np.ones(m), X]
    a1 = sigmoid(X @ Theta1.T)
    a1 = np.c_[np.ones(a1.shape[0]), a1]
    a2 = sigmoid(a1 @ Theta2.T)
    p = np.argmax(a2, axis=1) + 1

    return p