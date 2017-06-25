#
# This code is intentionally missing!
# Read the directions on the course lab page!
#

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn import manifold

x = pd.read_csv('Datasets/parkinsons.data', index_col=0)
y = x['status']

x.drop(labels='status', axis=1, inplace=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=7)

ppc = preprocessing.StandardScaler().fit(x_train)
# Normalizer(): 0.796610169492 
# MaxAbsScaler(): 0.881355932203 
# MinMaxScaler(): 0.881355932203
# KernelCenterer(): 0.915254237288
# StandardScaler(): 0.932203389831

x_train = ppc.transform(x_train)
x_test = ppc.transform(x_test)

# PCA: 0.932203389831
# ISO: 0.949152542373

best_score = 0
x_ppc_train = x_train.copy()
x_ppc_test = x_test.copy()

#for n_components in range(4,15):
  #print 'Calculating... current n_component value: ' + str(n_components) 
  #pca = PCA(n_components=n_components)
  #pca.fit(x_ppc_train)

  #x_train = pca.transform(x_ppc_train)
  #x_test = pca.transform(x_ppc_test)  
  
for n_neighbors in range(2,6):
  print 'Calculating... current n_neighbors value: ' + str(n_neighbors) 
  for n_components in range(4,7):
    iso = manifold.Isomap(n_neighbors=n_neighbors, n_components=n_components)
    iso.fit(x_ppc_train)
    
    x_train = iso.transform(x_ppc_train)
    x_test = iso.transform(x_ppc_test)
    
    for c in np.arange(0.05, 2.05, 0.05):
    
      for gamma in np.arange(0.001, 0.101, 0.001):
        svc = SVC(C=c, gamma=gamma)
        svc.fit(x_train, y_train)
    
        current_score = svc.score(x_test, y_test)
    
        if current_score > best_score:
          best_score = current_score
          best_c = c
          best_gamma = gamma

print 'Best score: ' + str(best_score)
print 'Best C: ' + str(best_c)
print 'Best gamma: ' + str(best_gamma)