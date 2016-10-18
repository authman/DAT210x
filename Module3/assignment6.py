# %load assignment6.py
import pandas as pd
import matplotlib.pyplot as plt

#
# TODO: Load up the Seeds Dataset into a Dataframe
# It's located at 'Datasets/wheat.data'
# 
# .. your code here ..
wheatCorr = pd.read_csv('Datasets/wheat.data')

#
# TODO: Drop the 'id' feature
# 
# .. your code here ..
wheatCorrDrop = wheatCorr.drop('id', axis = 1)

#
# TODO: Compute the correlation matrix of your dataframe
# 
# .. your code here ..
print(wheatCorrDrop.corr())

#
# TODO: Graph the correlation matrix using imshow or matshow
# 
# .. your code here ..

plt.imshow(wheatCorrDrop.corr(), cmap = plt.cm.Greens, interpolation= 'nearest')
plt.colorbar()
tick_marks = [i for i in range(len(wheatCorrDrop.columns))]
plt.xticks(tick_marks, wheatCorrDrop.columns, rotation = 'vertical')
plt.yticks(tick_marks, wheatCorrDrop.columns)


plt.show()

