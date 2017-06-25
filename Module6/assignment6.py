import pandas as pd
import numpy as np
import time


X = pd.read_csv('Datasets/ugulino.csv', delimiter=';')

X['gender'] = X['gender'].map({'Man': 0, 'Woman': 1})

X['how_tall_in_meters'] = X['how_tall_in_meters'].apply(lambda x: x.replace(',', '.'))
X['body_mass_index'] = X['body_mass_index'].apply(lambda x: x.replace(',', '.'))

X['how_tall_in_meters'] = pd.to_numeric(X['how_tall_in_meters'])
X['body_mass_index'] = pd.to_numeric(X['body_mass_index'])

ind = 0
inds = []
for row in X['z4']:
  #if type(row) == str:
  if row == '-14420-11-2011 04:50:23.713':
    inds.append(ind)
  ind+=1

X.drop(X.index[inds], inplace=True)

X['z4'] = pd.to_numeric(X['z4'], errors='raise')

print X.dtypes

Y = pd.get_dummies(X['class'])
print Y.head()

X.drop(labels=['user', 'class'], axis=1, inplace=True)
print X.describe()

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

model = RandomForestClassifier(n_estimators=30, max_depth=10, oob_score=True, random_state=0)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=7)

print "Fitting..."
s = time.time()
model.fit(X_train, Y_train)
print "Fitting completed in: ", time.time() - s
score = model.oob_score_
print "OOB Score: ", round(score*100, 3)

print "Scoring..."
s = time.time()
score = model.score(X_test, Y_test)
print "Score: ", round(score*100, 3)
print "Scoring completed in: ", time.time() - s


#
# TODO: Answer the lab questions, then come back to experiment more


#
# TODO: Try playing around with the gender column
# Encode it as Male:1, Female:0
# Try encoding it to pandas dummies
# Also try dropping it. See how it affects the score
# This will be a key on how features affect your overall scoring
# and why it's important to choose good ones.



#
# TODO: After that, try messing with 'y'. Right now its encoded with
# dummies try other encoding methods to experiment with the effect.

