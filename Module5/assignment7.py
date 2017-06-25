# If you'd like to try this lab with PCA instead of Isomap,
# as the dimensionality reduction technique:
Test_PCA = True

import numpy as np

def plotDecisionBoundary(model, X, y):
  print "Plotting..."
  import matplotlib.pyplot as plt
  import matplotlib
  matplotlib.style.use('ggplot') # Look Pretty

  fig = plt.figure()
  ax = fig.add_subplot(111)

  padding = 0.1
  resolution = 0.1

  #(2 for benign, 4 for malignant)
  colors = {2:'royalblue',4:'lightsalmon'} 

  
  # Calculate the boundaris
  x_min, x_max = X[:, 0].min(), X[:, 0].max()
  y_min, y_max = X[:, 1].min(), X[:, 1].max()
  x_range = x_max - x_min
  y_range = y_max - y_min
  x_min -= x_range * padding
  y_min -= y_range * padding
  x_max += x_range * padding
  y_max += y_range * padding

  # Create a 2D Grid Matrix. The values stored in the matrix
  # are the predictions of the class at at said location
  xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                       np.arange(y_min, y_max, resolution))

  # What class does the classifier say?
  Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)

  # Plot the contour map
  plt.contourf(xx, yy, Z, cmap=plt.cm.seismic)
  plt.axis('tight')

  # Plot your testing points as well...
  for label in np.unique(y):
    indices = np.where(y == label)
    plt.scatter(X[indices, 0], X[indices, 1], c=colors[label], alpha=0.8)

  p = model.get_params()
  plt.title('K = ' + str(p['n_neighbors']))
  plt.show()

import pandas as pd

names = ['thickness', 'size', 'shape', 'adhesion', 'epithelial', 'nuclei', 'chromatin', 'nucleoli', 'mitoses', 'status']
df = pd.read_csv('Datasets/breast-cancer-wisconsin.data', index_col=0, names=names)

labels = df['status']
df.drop(labels=['status'], inplace=True, axis = 1)

df.replace('?', np.NaN, inplace=True)
df['nuclei'] = pd.to_numeric(df['nuclei'], errors='coerce')

df = df.fillna(df.mean(axis=0))

from sklearn.model_selection import train_test_split

data_train, data_test, label_train, label_test = train_test_split(df, labels, test_size=0.5, random_state=7)

from sklearn import preprocessing

preprocessor = preprocessing.Normalizer().fit(data_train)
# MaxAbsScaler(), MinMaxScaler(), StandardScaler(), Normalizer(), RobustScaler()

data_train = preprocessor.transform(data_train)
data_test = preprocessor.transform(data_test)

#print data_train.shape

#
# PCA and Isomap are your new best friends
model = None
if Test_PCA:
  print "Computing 2D Principle Components"
  
  from sklearn.decomposition import PCA
  model = PCA(n_components=2)

else:
  print "Computing 2D Isomap Manifold"
  
  from sklearn import manifold
  model = manifold.Isomap(n_neighbors=5, n_components=2)


model = model.fit(data_train)
data_train = model.transform(data_train)
data_test = model.transform(data_test)  

from sklearn.neighbors import KNeighborsClassifier

for k in range(1,16):
  knmodel = KNeighborsClassifier(n_neighbors=k, weights='distance')
  knmodel.fit(data_train, label_train)
  
  score = knmodel.score(data_test, label_test)
  print score

# 
# TODO: Implement and train KNeighborsClassifier on your projected 2D
# training data here. You can use any K value from 1 - 15, so play around
# with it and see what results you can come up. Your goal is to find a
# good balance where you aren't too specific (low-K), nor are you too
# general (high-K). You should also experiment with how changing the weights
# parameter affects the results.
#
# .. your code here ..

#
# INFO: Be sure to always keep the domain of the problem in mind! It's
# WAY more important to errantly classify a benign tumor as malignant,
# and have it removed, than to incorrectly leave a malignant tumor, believing
# it to be benign, and then having the patient progress in cancer. Since the UDF
# weights don't give you any class information, the only way to introduce this
# data into SKLearn's KNN Classifier is by "baking" it into your data. For
# example, randomly reducing the ratio of benign samples compared to malignant
# samples from the training set.

plotDecisionBoundary(knmodel, data_test, label_test)
