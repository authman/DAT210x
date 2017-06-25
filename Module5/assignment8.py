import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use('ggplot') # Look Pretty


def drawLine(model, X_test, y_test, title):
  # This convenience method will take care of plotting your
  # test observations, comparing them to the regression line,
  # and displaying the R2 coefficient
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.scatter(X_test, y_test, c='g', marker='o')
  ax.plot(X_test, model.predict(X_test), color='orange', linewidth=1, alpha=0.7)

  print "Est 2014 " + title + " Life Expectancy: ", model.predict([[2014]])[0]
  print "Est 2030 " + title + " Life Expectancy: ", model.predict([[2030]])[0]
  print "Est 2045 " + title + " Life Expectancy: ", model.predict([[2045]])[0]

  score = model.score(X_test, y_test)
  title += " R2: " + str(score)
  ax.set_title(title)

  plt.show()

headers = ['Year', 'WhiteMale', 'WhiteFemale', 'BlackMale', 'BlackFemale']
X = pd.read_csv('Datasets/life_expectancy.csv', delim_whitespace=True, skiprows=1, names=headers)
print len(X)

def interpolate(rase_and_gender):
  from sklearn import linear_model
  model = linear_model.LinearRegression()

  X_train = X[X['Year'] < 1986][['Year']]
  Y_train = X[X['Year'] < 1986][rase_and_gender]

  X_test = X[X['Year'] >= 1986][['Year']]
  Y_test = X[X['Year'] >= 1986][rase_and_gender]

  model.fit(X_train, Y_train)

  drawLine(model, X_test, Y_test, rase_and_gender)

  print X[X['Year'] == 2014]

interpolate('WhiteMale')
#interpolate('BlackFemale')

print X.corr()
plt.imshow(X.corr(), cmap=plt.cm.Blues, interpolation='nearest')
plt.colorbar()
tick_marks = [i for i in range(len(X.columns))]
plt.xticks(tick_marks, X.columns, rotation='vertical')
plt.yticks(tick_marks, X.columns)

plt.show()

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

