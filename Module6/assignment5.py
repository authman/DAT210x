import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import tree

#https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.names

headers = ['classification', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor'
            'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 
            'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 
            'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color',
            'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']
X = pd.read_csv('Datasets/agaricus-lepiota.data', names=headers, index_col=False)

print len(X)

X.replace('?', np.NaN, inplace=True)
# INFO: An easy way to show which rows have nans in them
#print X[pd.isnull(X).any(axis=1)]
X.dropna(inplace=True, axis=0)
#print X[pd.isnull(X).any(axis=1)]

print len(X)

Y = X['classification']
X.drop(labels='classification', axis=1, inplace=True)
Y = Y.map({'p':0, 'e':1})

#print X.head()
#print Y.head()

X = pd.get_dummies(X)

#print X.head()
#print list(X.columns.values)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=7)

dtree = tree.DecisionTreeClassifier()
dtree.fit(X_train, Y_train)

score = dtree.score(X_test, Y_test)

print "High-Dimensionality Score: ", round((score*100), 3)

fi_ind=0
for fi in dtree.feature_importances_:
  #print str(fi_ind) + ' ===> ' + str(fi)
  fi_ind+=1

#print dtree.classes_
tree.export_graphviz(dtree.tree_, out_file='tree.dot', feature_names=X.columns)

from subprocess import call
call(['dot', '-T', 'png', 'tree.dot', '-o', 'tree.png'])