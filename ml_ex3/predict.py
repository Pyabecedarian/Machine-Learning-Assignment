import numpy as np

from ml_ex3.sigmoid import sigmoid


def predict(Theta1, Theta2, X):
    """
    PREDICT Predict the label of an input given a trained neural network
       p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
       trained weights of a neural network (Theta1, Theta2)

    NOTE:
        Theta1.shape = (25, 401)     Theta2.shape = (10, 26)     X.shape = (5000, 400)
    """
    # % Useful values
    m, _ = X.shape
    num_labels, _ = Theta2.shape

    # % You need to return the following variables correctly
    # p = np.zeros(m)

    # % ====================== YOUR CODE HERE ======================
    # % Instructions: Complete the following code to make predictions using
    # %               your learned neural network. You should set p to a
    # %               vector containing labels between 1 to num_labels.
    # %
    # % Hint: The max function might come in useful. In particular, the max
    # %       function can also return the index of the max element, for more
    # %       information see 'help max'. If your examples are in rows, then, you
    # %       can use max(A, [], 2) to obtain the max for each row.
    # %
    #
    """
    NOTE:
    
             Layer1                               Layer2                               Layer3
               X          -----θ1-----              a2        -------θ2-------           a3
          (5000, 400)       (25, 401)              (,)             (10, 26)             (10,)
                        
        add 1's column  >>>  θ1 @ X.T     =    a2, (25, 5000)     
                                           a2.T, add 1's column
                                                (5000, 26)     >>>  θ2 @ a2.T      =     a3 (10, 5000)                                  
    """
    # Layer1 to Layer2
    X = np.hstack((np.ones(m).reshape(m, 1), X))
    a2 = sigmoid(Theta1 @ X.T)

    # Layer2 to Layer3
    _, n = a2.shape
    a2 = np.hstack((np.ones(n).reshape(n, 1), a2.T))
    a3 = sigmoid(Theta2 @ a2.T)

    # pred should be the maximum of each row
    pred = np.argmax(a3, axis=0) + 1  # return the max row num of each column
    # % =========================================================================
    return pred
