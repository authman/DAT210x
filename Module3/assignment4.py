# %load assignment4.py
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

from pandas.tools.plotting import parallel_coordinates

# Look pretty...
matplotlib.style.use('ggplot')


#
# TODO: Load up the Seeds Dataset into a Dataframe
# It's located at 'Datasets/wheat.data'
# 
# .. your code here ..
wheatParl = pd.read_csv('Datasets/wheat.data')



#
# TODO: Drop the 'id', 'area', and 'perimeter' feature
# 
# .. your code here ..

wheatDrop = wheatParl.drop(['id', 'area', 'perimeter'], axis = 1)


#
# TODO: Plot a parallel coordinates chart grouped by
# the 'wheat_type' feature. Be sure to set the optional
# display parameter alpha to 0.4
# 
# .. your code here ..
plt.figure()
parallel_coordinates(wheatDrop, 'wheat_type') #remember the features needs to be normalized before you plot
#And often 'parallel_coordinates' are not very sweet with more than 10 features

plt.show()




#?matplotlib.style.use() A very good way to ask for help on a particular part of documentation in ipython