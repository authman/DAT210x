import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.style.use('ggplot') # Look Pretty


def plotDecisionBoundary(model, X, y):
  fig = plt.figure()
  ax = fig.add_subplot(111)

  padding = 0.6
  resolution = 0.0025
  colors = ['royalblue','forestgreen','ghostwhite']
  
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
  cs = plt.contourf(xx, yy, Z, cmap=plt.cm.terrain)

  # Plot the test original points as well...
  for label in range(len(np.unique(y))):
    indices = np.where(y == label)
    plt.scatter(X[indices, 0], X[indices, 1], c=colors[label], label=str(label), alpha=0.8)

  p = model.get_params()
  plt.axis('tight')
  plt.title('K = ' + str(p['n_neighbors']))

X = pd.read_csv('Datasets/wheat.data', index_col=0)
print X.head()

#print 'Max ===> ' + str(X[:,0].max())

Y = X['wheat_type'].copy()
X.drop(labels=['wheat_type'], inplace=True, axis=1)

conversion_dict = {'kama':1, 'canadian':2, 'rosa':3}
Y = Y.apply(conversion_dict.get)

print Y.head()
print Y.unique()
print X.mean()

X = X.fillna(X.mean())

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.decomposition import PCA

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=1)

normalizer = preprocessing.Normalizer().fit(X_train)#, Y_train)
normalized_X_train = normalizer.transform(X_train)
normalized_X_test = normalizer.transform(X_test)

pca = PCA(n_components=2, svd_solver='full').fit(normalized_X_train)#, Y_train)
pca_X_train = pca.transform(normalized_X_train)
pca_X_test = pca.transform(normalized_X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(pca_X_train, Y_train)

plotDecisionBoundary(knn, pca_X_train, Y_train)

print knn.score(pca_X_test, Y_test)

#
# BONUS: Instead of the ordinal conversion, try and get this assignment
# working with a proper Pandas get_dummies for feature encoding. HINT:
# You might have to update some of the plotDecisionBoundary code.

plt.show()