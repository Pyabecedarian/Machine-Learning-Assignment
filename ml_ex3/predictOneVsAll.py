import numpy as np

from ml_ex3.sigmoid import sigmoid


def predictOneVsAll(all_theta, X):
    """
    PREDICT Predict the label for a trained one-vs-all classifier. The labels
    are in the range 1..K, where K = size(all_theta, 1).
      p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
      for each example in the matrix X. Note that X contains the examples in
      rows. all_theta is a matrix where the i-th row is a trained logistic
      regression theta vector for the i-th class. You should set p to a vector
      of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
      for 4 examples)
    """

    m, _ = X.shape
    num_labels, _ = all_theta.shape

    # % You need to return the following variables correctly
    p = np.zeros(m)

    # % Add ones to the X data matrix
    X = np.hstack((np.ones(m).reshape(m, 1), X))

    # % ====================== YOUR CODE HERE ======================
    # % Instructions: Complete the following code to make predictions using
    # %               your learned logistic regression parameters (one-vs-all).
    # %               You should set p to a vector of predictions (from 1 to
    # %               num_labels).
    # %
    # % Hint: This code can be done all vectorized using the max function.
    # %       In particular, the max function can also return the index of the
    # %       max element, for more information see 'help max'. If your examples
    # %       are in rows, then, you can use max(A, [], 2) to obtain the max
    # %       for each row.
    # %
    """
    NOTE:
        If m is the NO. of predictions in X,  X is m×N matrix (each prediction contains N features);     
        K is the NO. of classifiers, then:
         
                               [  h(K=1, 1),  h(K=1, 2),  h(K=1, 3), ... ... ... h(K=1, m)  ]
                               [  h(K=2, 1),  h(K=2, 2),  h(K=2, 3), ... ... ... h(K=2, m)  ]
                               [     .          ...          ...                    .       ]
        H = g( θ @ X.T ) =     [     .                                              .       ]
                               [     .          ...          ...                    .       ]
                               [     .                                              .       ]
                               [  h(K=K, 1),  h(K=K, 2),  h(K=K, 3), ... ... ... h(K=K, m)  ]
    """
    Z = all_theta @ X.T
    H = sigmoid(Z)  # This line can be omitted but recommended here as it gives the probability of each prediction

    # NOTE: The predicted num should be "row indices + 1"
    pred = np.argmax(H, axis=0) + 1   # Find row num of the max in each column.
    # % =========================================================================

    return pred
