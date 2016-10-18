# %load assignment5.py
#
# This code is intentionally missing!
# Read the directions on the course lab page!
#
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

#import andrews_curves
from pandas.tools.plotting import andrews_curves

#Read in the dataset
wheatAndrew = pd.read_csv('Datasets/wheat.data')

#Drop 'id', 'area' and 'perimeter'
wheatADrop = wheatAndrew.drop(['id', 'area', 'perimeter'], axis = 1)
wheatADrop.head()

#Prepare Andrew plot for the data
plt.figure()
andrews_curves(wheatADrop, 'wheat_type')
plt.show()


#Add back 'area' and 'perimeter'
wheatComp = pd.concat([wheatADrop, wheatAndrew.area,  wheatAndrew.perimeter], axis = 1)

#Plot Andrew Curve
plt.figure()
andrews_curves(wheatComp, 'wheat_type')
plt.show()
#With the Addition of the other features 'area' and 'perimeter' the feature scaling is no more feasible