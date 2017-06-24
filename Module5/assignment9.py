import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

matplotlib.style.use('ggplot') # Look Pretty


def drawLine(model, X_test, y_test, title, R2):
  # This convenience method will take care of plotting your
  # test observations, comparing them to the regression line,
  # and displaying the R2 coefficient
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.scatter(X_test, y_test, c='g', marker='o')
  ax.plot(X_test, model.predict(X_test), color='orange', linewidth=1, alpha=0.7)

  title += " R2: " + str(R2)
  ax.set_title(title)
  print title
  print "Intercept(s): ", model.intercept_

  plt.show()

def drawPlane(model, X_test, y_test, title, R2):
  # This convenience method will take care of plotting your
  # test observations, comparing them to the regression plane,
  # and displaying the R2 coefficient
  fig = plt.figure()
  ax = Axes3D(fig)
  ax.set_zlabel('prediction')

  # You might have passed in a DataFrame, a Series (slice),
  # an NDArray, or a Python List... so let's keep it simple:
  X_test = np.array(X_test)
  col1 = X_test[:,0]
  col2 = X_test[:,1]

  # Set up a Grid. We could have predicted on the actual
  # col1, col2 values directly; but that would have generated
  # a mesh with WAY too fine a grid, which would have detracted
  # from the visualization
  x_min, x_max = col1.min(), col1.max()
  y_min, y_max = col2.min(), col2.max()
  x = np.arange(x_min, x_max, (x_max-x_min) / 10)
  y = np.arange(y_min, y_max, (y_max-y_min) / 10)
  x, y = np.meshgrid(x, y)

  # Predict based on possible input values that span the domain
  # of the x and y inputs:
  z = model.predict(  np.c_[x.ravel(), y.ravel()]  )
  z = z.reshape(x.shape)

  ax.scatter(col1, col2, y_test, c='g', marker='o')
  ax.plot_wireframe(x, y, z, color='orange', alpha=0.7)
  
  title += " R2: " + str(R2)
  ax.set_title(title)
  print title
  print "Intercept(s): ", model.intercept_
  
  plt.show()
  
X = pd.read_csv('Datasets/College.csv', index_col=0)
#print X.head()

X.Private = X.Private.map({'Yes':1, 'No':0})

def score_relationship(data_column, draw='line'):
  from sklearn import linear_model
  model = linear_model.LinearRegression()

  data = X[data_column]
  labels = X['Accept']

  from sklearn.model_selection import train_test_split
  X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.3, random_state=7)

  model.fit(X_train, Y_train)

  score = model.score(X_test, Y_test)

  # INFO: We'll take it from here, buddy:
  if (draw=='line'):
    drawLine(model, X_test, Y_test, "Accept(" + str(data_column) + ")", score)
  else:
    drawPlane(model, X_test, Y_test, "Accept(" + str(data_column) + ")", score)

score_relationship(['Room.Board'])
score_relationship(['Enroll'])
score_relationship(['F.Undergrad'])
score_relationship(['Room.Board', 'Enroll'], 'plane')

#
# TODO: Duplicate the process above (almost). This time is going to be
# a bit more complicated. Instead of modeling one feature as a function
# of another, you will attempt to do multivariate linear regression to
# model one feature as a function of TWO other features.
#
# Model the amount charged for room and board AND the number of enrolled
# students, as a function of the number of accepted students. To do
# this, instead of creating a regular slice for a single-feature input,
# simply create a slice that contains both columns you wish to use as
# inputs. Your training labels will remain a single slice.
#
# .. your code here ..


#
# INFO: That concludes this assignment
#




# INFO + HINT On Fitting, Scoring, and Predicting:
#
# Here's a hint to help you complete the assignment without pulling
# your hair out! When you use .fit(), .score(), and .predict() on
# your model, SciKit-Learn expects your training data to be in
# spreadsheet (2D Array-Like) form. This means you can't simply
# pass in a 1D Array (slice) and get away with it.
#
# To properly prep your data, you have to pass in a 2D Numpy Array,
# or a dataframe. But what happens if you really only want to pass
# in a single feature?
#
# If you slice your dataframe using df[['ColumnName']] syntax, the
# result that comes back is actually a *dataframe*. Go ahead and do
# a type() on it to check it out. Since it's already a dataframe,
# you're good -- no further changes needed.
#
# But if you slice your dataframe using the df.ColumnName syntax,
# OR if you call df['ColumnName'], the result that comes back is
# actually a series (1D Array)! This will cause SKLearn to bug out.
# So if you are slicing using either of those two techniques, before
# sending your training or testing data to .fit / .score, do a
# my_column = my_column.reshape(-1,1). This will convert your 1D
# array of [n_samples], to a 2D array shaped like [n_samples, 1].
# A single feature, with many samples.
#
# If you did something like my_column = [my_column], that would produce
# an array in the shape of [1, n_samples], which is incorrect because
# SKLearn expects your data to be arranged as [n_samples, n_features].
# Keep in mind, all of the above only relates to your "X" or input
# data, and does not apply to your "y" or labels.





#
# Data Scientist Challenge
# ========================
#
# You've experimented with a number of feature scaling techniques
# already, such as MaxAbsScaler, MinMaxScaler, Normalizer, StandardScaler
# and more from http://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
# 
# What happens if you apply scaling to your data before doing 
# linear regression? Would it alter the quality of your results?
# Do the scalers that work on a per-feature basis, such as MinMaxScaler
# behave differently that those that work on a multi-feature basis, such
# as normalize? And moreover, once your features have been scaled, you
# won't be able to use the resulting regression directly... unless you're
# able to .inverse_transform() the scaling. Do all of the SciKit-Learn
# scalers support that?
#
# This is your time to shine and to show how much of an explorer you are:
# Dive deeper into uncharted lands, browse SciKit-Learn's documentation,
# scour Google, ask questions on Quora, Stack-Overflow, and the course
# message board, and see if you can discover something that will be of
# benefit to you in the future!


