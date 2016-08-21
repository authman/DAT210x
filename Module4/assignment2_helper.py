import math
import pandas as pd
from sklearn import preprocessing

# A Note on SKLearn .transform() calls:
#
# Any time you transform your data, you lose the column header names.
# This actually makes complete sense. There are essentially two types
# of transformations,  those that change the scale of your features,
# and those that change your features entire. Changing the scale would
# be like changing centimeters to inches. Changing the features would
# be like using PCA to reduce 300 columns to 30. In either case, the
# original column's units have been altered or no longer exist, so it's
# up to you to rename your columns after ANY transformation. Due to
# this, SKLearn returns an NDArray from *transform() calls.

def scaleFeatures(df):
  # SKLearn has many different methods for doing transforming your
  # features by scaling them (this is a type of pre-processing).
  # RobustScaler, Normalizer, MinMaxScaler, MaxAbsScaler, StandardScaler...
  # http://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
  #
  # However in order to be effective at PCA, there are a few requirements
  # that must be met, and which will drive the selection of your scaler.
  # PCA required your data is standardized -- in other words it's mean is
  # equal to 0, and it has ~unit variance.
  #
  # SKLearn's regular Normalizer doesn't zero out the mean of your data,
  # it only clamps it, so it's inappropriate to use here (depending on
  # your data). MinMaxScaler and MaxAbsScaler both fail to set a unit
  # variance, so you won't be using them either. RobustScaler can work,
  # again depending on your data (watch for outliers). For these reasons
  # we're going to use the StandardScaler. Get familiar with it by visiting
  # these two websites:
  #
  # http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-scaler
  #
  # http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler
  #


  # ---------
  # Feature scaling is the type of transformation that only changes the
  # scale and not number of features, so we'll use the original dataset
  # column names. However we'll keep in mind that the _units_ have been
  # altered:
  scaled = preprocessing.StandardScaler().fit_transform(df)
  scaled = pd.DataFrame(scaled, columns=df.columns)
  print "New Variances:\n", scaled.var()
  print "New Describe:\n", scaled.describe()
  return scaled


def drawVectors(transformed_features, components_, columns, plt, scaled):
  if not scaled:
    return plt.axes() # No cheating ;-)

  num_columns = len(columns)

  # This funtion will project your *original* feature (columns)
  # onto your principal component feature-space, so that you can
  # visualize how "important" each one was in the
  # multi-dimensional scaling
  
  # Scale the principal components by the max value in
  # the transformed set belonging to that component
  xvector = components_[0] * max(transformed_features[:,0])
  yvector = components_[1] * max(transformed_features[:,1])

  ## visualize projections

  # Sort each column by it's length. These are your *original*
  # columns, not the principal components.
  important_features = { columns[i] : math.sqrt(xvector[i]**2 + yvector[i]**2) for i in range(num_columns) }
  important_features = sorted(zip(important_features.values(), important_features.keys()), reverse=True)
  print "Features by importance:\n", important_features

  ax = plt.axes()

  for i in range(num_columns):
    # Use an arrow to project each original feature as a
    # labeled vector on your principal component axes
    plt.arrow(0, 0, xvector[i], yvector[i], color='b', width=0.0005, head_width=0.02, alpha=0.75)
    plt.text(xvector[i]*1.2, yvector[i]*1.2, list(columns)[i], color='b', alpha=0.75)

  return ax
    