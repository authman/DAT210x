# %load assignment1.py
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
#%matplotlib inline

# Look pretty...
matplotlib.style.use('ggplot')

#
# TODO: Load up the Seeds Dataset into a Dataframe
# It's located at 'Datasets/wheat.data'
# 
# .. your code here ..
wheatData = pd.read_csv('Datasets/wheat.data')
wheatData.head()
#Do some check on the dataframe
print(wheatData.isnull().any())
print(wheatData.dtypes)

#After observing that there are null values, I tried to check if its only null and not unclean data that are in dataset
#for i, v in enumerate(wheatData):
#    print(wheatData[v].unique())

#
# TODO: Create a slice of your dataframe (call it s1)
# that only includes the 'area' and 'perimeter' features
# 
# .. your code here ..
s1 = wheatData.loc[:, ['area', 'perimeter']]


#
# TODO: Create another slice of your dataframe (call it s2)
# that only includes the 'groove' and 'asymmetry' features
# 
# .. your code here ..
s2 = wheatData.loc[:, ['groove', 'asymmetry']]

#
# TODO: Create a histogram plot using the first slice,
# and another histogram plot using the second slice.
# Be sure to set alpha=0.75
# 
# .. your code here ..
ax = s1.plot.hist(alpha = 0.75)
ax2 = s2.plot.hist(alpha = 0.75)
ax.set_xlabel("Values")
ax2.set_xlabel("Values") #if you want to set the x and y labels for a pandas plot

plt.show()
