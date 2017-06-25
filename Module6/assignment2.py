import matplotlib as mpl
import matplotlib.pyplot as plt

import pandas as pd

# The Dataset comes from:
# https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits

# At face value, this looks like an easy lab;
# But it has many parts to it, so prepare yourself before starting...


def load(path_test, path_train):
  # Load up the data.
  # You probably could have written this..
  with open(path_test, 'r')  as f: testing  = pd.read_csv(f)
  with open(path_train, 'r') as f: training = pd.read_csv(f)

  # The number of samples between training and testing can vary
  # But the number of features better remain the same!
  n_features = testing.shape[1]

  X_test  = testing.ix[:,:n_features-1]
  X_train = training.ix[:,:n_features-1]
  y_test  = testing.ix[:,n_features-1:].values.ravel()
  y_train = training.ix[:,n_features-1:].values.ravel()

  #
  # Special:

  return X_train, X_test, y_train, y_test


def peekData(X_train):
  # The 'targets' or labels are stored in y. The 'samples' or data is stored in X
  print "Peeking your data..."
  fig = plt.figure()

  cnt = 0
  for col in range(5):
    for row in range(10):
      plt.subplot(5, 10, cnt + 1)
      plt.imshow(X_train.ix[cnt,:].reshape(8,8), cmap=plt.cm.gray_r, interpolation='nearest')
      plt.axis('off')
      cnt += 1
  fig.set_tight_layout(True)
  plt.show()


def drawPredictions(X_train, X_test, y_train, y_test):
  fig = plt.figure()

  # Make some guesses
  y_guess = model.predict(X_test)


  #
  # INFO: This is the second lab we're demonstrating how to
  # do multi-plots using matplot lab. In the next assignment(s),
  # it'll be your responsibility to use this and assignment #1
  # as tutorials to add in the plotting code yourself!
  num_rows = 10
  num_cols = 5

  index = 0
  for col in range(num_cols):
    for row in range(num_rows):
      plt.subplot(num_cols, num_rows, index + 1)

      # 8x8 is the size of the image, 64 pixels
      plt.imshow(X_test.ix[index,:].reshape(8,8), cmap=plt.cm.gray_r, interpolation='nearest')

      # Green = Guessed right
      # Red = Fail!
      fontcolor = 'g' if y_test[index] == y_guess[index] else 'r'
      plt.title('Label: %i' % y_guess[index], fontsize=6, color=fontcolor)
      plt.axis('off')
      index += 1
  fig.set_tight_layout(True)
  plt.show()

X_train, X_test, Y_train, Y_test = load('Datasets/optdigits.tes', 'Datasets/optdigits.tra')

from sklearn import svm

# Get to know your data. It seems its already well organized in
# [n_samples, n_features] form. Our dataset looks like (4389, 784).
# Also your labels are already shaped as [n_samples].
#peekData(X_train)

from sklearn.svm import SVC
model = SVC(C=1.0, gamma=0.001, kernel='rbf')
print "Training SVC Classifier..."
model.fit(X_train, Y_train)

print "Scoring SVC Classifier..."
score = model.score(X_test, Y_test)
print "Score:\n", score

# Visual Confirmation of accuracy
#drawPredictions(X_train, X_test, Y_train, Y_test)

true_1000th_test_value = Y_test[999]
true_1001st_test_value = Y_test[1000]
print "1000th test label: ", true_1000th_test_value
print "1001st test label: ", true_1001st_test_value

guess_1000th_test_value = model.predict(X_test)
print "1000th test prediction: ", guess_1000th_test_value[999]
print "1001st test prediction: ", guess_1000th_test_value[1000]

plt.subplot(2, 1, 1)
plt.imshow(X_test.ix[999,:].reshape(8,8), cmap=plt.cm.gray_r, interpolation='nearest')
plt.subplot(2, 1, 2)
plt.imshow(X_test.ix[1000,:].reshape(8,8), cmap=plt.cm.gray_r, interpolation='nearest')

# linear: 0.961024498886
# poly: 0.974944320713
# rbf: 0.982739420935
    

#
# TODO: Once you're able to beat the +98% accuracy score of the
# USPS, go back into the load() method. Look for the line that
# reads "# Special:"
#
# Immediately under that line, alter X_train and y_train ONLY.
# Keep just the ___FIRST___ 4% of the samples. In other words,
# for every 100 samples found, throw away 96 of them. Make sure
# all the samples (and labels) you keep come from the start of
# X_train and y_train.

# If the first 4% is a decimal number, then use int + ceil to
# round up to the nearest whole integer.

# That operation might require some Pandas indexing skills, or
# perhaps some numpy indexing skills if you'd like to go that
# route. Feel free to ask on the class forum if you want; but
# try to exercise your own muscles first, for at least 30
# minutes, by reviewing the Pandas documentation and stack
# overflow. Through that, in the process, you'll pick up a lot.
# Part of being a machine learning practitioner is know what
# questions to ask and where to ask them, so this is a great
# time to start!

# Re-Run your application after throwing away 96% your training
# data. What accuracy score do you get now?




#
# TODO: Lastly, change your kernel back to linear and run your
# assignment one last time. What's the accuracy score this time?
# Surprised?

