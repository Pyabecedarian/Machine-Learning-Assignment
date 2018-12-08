"""
%% Machine Learning Online Class
%  Exercise 6 | Spam Classification with SVMs
%
%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the
%  exercise. You will need to complete the following functions:
%
%     gaussianKernel.m
%     dataset3Params.m
%     processEmail.m
%     emailFeatures.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%
"""
import numpy as np
import scipy.io as sio
from sklearn import svm

from ml_ex6.processEmail import processEmail
from ml_ex6.emailFeatures import emailFeatures
from ml_ex6.getVocabList import getVocabList

print('p1', '---' * 20)
# %% ==================== Part 1: Email Preprocessing ====================
# %  To use an SVM to classify emails into Spam v.s. Non-Spam, you first need
# %  to convert each email into a vector of features. In this part, you will
# %  implement the preprocessing steps for each email. You should
# %  complete the code in processEmail.m to produce a word indices vector
# %  for a given email.

print('Preprocessing sample email (emailSample1.txt)')
# % Extract Features
with open('emailSample1.txt') as f_obj:
    file_contents = f_obj.read()
word_indices = processEmail(file_contents)

# % Print Stats
print('Word Indices:')
print(word_indices)
print('\n\n')
print('Program paused. Press enter to continue.\n')

print('p2', '---' * 20)
# %% ==================== Part 2: Feature Extraction ====================
# %  Now, you will convert each email into a vector of features in R^n.
# %  You should complete the code in emailFeatures.m to produce a feature
# %  vector for a given email.
print('Extracting features from sample email (emailSample1.txt)')
# % Extract Features
features = emailFeatures(word_indices)

# Print Stats
print('Length of feature vector: %d' % features.size)
print('Number of non-zero entries: %d' % np.sum(features > 0))
print('Program paused. Press enter to continue.\n')

print('p3', '---' * 20)
# %% =========== Part 3: Train Linear SVM for Spam Classification ========
# %  In this section, you will train a linear classifier to determine if an
# %  email is Spam or Not-Spam.

# % Load the Spam Email dataset
# % You will have X, y in your environment
Data = sio.loadmat('spamTrain.mat')
X = Data['X']  # shape = (4000, 1899)
y = Data['y'].flatten()  # shape = (4000, )

print('\nTraining Linear SVM (Spam Classification)')
print('(this may take 1 to 2 minutes) ...\n')
C = 0.1
clf = svm.SVC(kernel='linear', C=C)
clf.fit(X, y)
p = clf.predict(X)
print('Training Accuracy: %f\n' % np.mean(p == y))
print('Program paused. Press enter to continue.\n')

print('p4', '---' * 20)
# %% =================== Part 4: Test Spam Classification ================
# %  After training the classifier, we can evaluate it on a test set. We have
# %  included a test set in spamTest.mat
#
# % Load the test dataset
# % You will have Xtest, ytest in your environment
Data = sio.loadmat('spamTest.mat')
Xtest = Data['Xtest']             # shape = (1000, 1899)
ytest = Data['ytest'].flatten()
print('Evaluating the trained Linear SVM on a test set ...\n')
p = clf.predict(Xtest)
print('Test Accuracy: %f\n' % clf.score(Xtest, ytest))
print('Program paused. Press enter to continue.\n')

print('p5', '---' * 20)
# %% ================= Part 5: Top Predictors of Spam ====================
# %  Since the model we are training is a linear SVM, we can inspect the
# %  weights learned by the model to understand better how it is determining
# %  whether an email is spam or not. The following code finds the words with
# %  the highest weights in the classifier. Informally, the classifier
# %  'thinks' that these words are the most likely indicators of spam.
# %

# % Sort the weights and obtain the vocabulary list
coef = clf.coef_.ravel()
top_positive_coefficients_index =np.flip(np.argsort(coef)[-15:])
top_positive_coefficients = coef[top_positive_coefficients_index]

vocabList = getVocabList()
print('Top predictors of spam:')
for index, tpci in enumerate(top_positive_coefficients_index):
    print('{} --- {:.4f}'.format(vocabList[tpci], top_positive_coefficients[index]))
print('Program paused. Press enter to continue.\n')

print('p6', '---' * 20)
# %% =================== Part 6: Try Your Own Emails =====================
# %  Now that you've trained the spam classifier, you can use it on your own
# %  emails! In the starter code, we have included spamSample1.txt,
# %  spamSample2.txt, emailSample1.txt and emailSample2.txt as examples.
# %  The following code reads in one of these emails and then uses your
# %  learned SVM classifier to determine whether the email is Spam or
# %  Not Spam
#
# % Set the file to be read in (change this to spamSample2.txt,
# % emailSample1.txt or emailSample2.txt to see different predictions on
# % different emails types). Try your own emails as well!
test_file = 'spamSample2.txt'
with open(test_file) as f_obj:
    file_contents = f_obj.read()
word_indices = processEmail(file_contents)
x = emailFeatures(word_indices)  # shape = (1899, )
# NOTE: To predict single example with n_features, x should be reshaped as shape = (1, -1)
# i.e.  x.shape = (1, 1899)
x = x.reshape(1, -1)
# Predict x whether or not is a spam email
p = clf.predict(x)
print('\n\nProcessed %s\nSpam Classification: %d\n' % (test_file, p[0]))
print('(1 indicates spam, 0 indicates not spam)\n\n')
print('Result: {} is a {}'.format(test_file, 'Spam Email' if p[0] else 'Normal Email'))
