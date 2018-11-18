"""
%% Machine Learning Online Class - Exercise 3 | Part 2: Neural Networks

%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the
%  linear exercise. You will need to complete the following functions
%  in this exericse:
%
%     lrCostFunction.m (logistic regression cost function)
%     oneVsAll.m
%     predictOneVsAll.m
%     predict.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%
"""
import scipy.io as sio
import numpy as np
from ml_ex3.displayData import displayData
from ml_ex3.predict import predict


# %% Setup the parameters you will use for this exercise
input_layer_size = 400    # % 20x20 Input Images of Digits
hidden_layer_size = 25    # % 25 hidden units
num_labels = 10           # % 10 labels, from 1 to 10
                          # % (note that we have mapped "0" to label 10)

# %% =========== Part 1: Loading and Visualizing Data =============
# %  We start the exercise by first loading and visualizing the dataset.
# %  You will be working with a dataset that contains handwritten digits.
# %

# % Load Training Data
data_dict = sio.loadmat('ex3data1.mat')
X = data_dict['X']
y = data_dict['y'].flatten()

m, _ = X.shape

# % Randomly select 100 data points to display
random_sel = np.random.permutation(m)
r_indices = random_sel[:100]
displayData(X[r_indices, :])


print('Program paused. Press enter to continue.\n')


# %% ================ Part 2: Loading Pameters ================
# % In this part of the exercise, we load some pre-initialized
# % neural network parameters.

print('\nLoading Saved Neural Network Parameters ...\n')

# % Load the weights into variables Theta1 and Theta2
weights_dict = sio.loadmat('ex3weights.mat')
Theta1 = weights_dict['Theta1']  # Theta1.shape = (25, 401)
Theta2 = weights_dict['Theta2']  # Theta2.shape = (10, 26)


# %% ================= Part 3: Implement Predict =================
# %  After training the neural network, we would like to use it to predict
# %  the labels. You will now implement the "predict" function to use the
# %  neural network to predict the labels of the training set. This lets
# %  you compute the training set accuracy.

pred = predict(Theta1, Theta2, X)
print('\nTraining Set Accuracy: %f\n' % (np.mean((pred == y) * 1.0)))

# print('Program paused. Press enter to continue.\n')
