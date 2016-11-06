# %load assignment5.py
import pandas as pd

import glob

from scipy import misc
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt

# Look pretty...
matplotlib.style.use('ggplot')


#
# TODO: Start by creating a regular old, plain, "vanilla"
# python list. You can call it 'samples'.
#
# .. your code here .. 
samples = [ ]
colors = []

for filename in glob.glob(r'Datasets/ALOI/32/*.png'):
    image = misc.imread(filename)
    image = image.reshape(-1)
    samples.append(image)
    colors.append('b')
for filename in glob.glob(r'Datasets/ALOI/32i/*.png'):
    image = misc.imread(filename)
    image = image.reshape(-1)
    samples.append(image)
    colors.append('r')
#
# TODO: Write a for-loop that iterates over the images in the
# Module4/Datasets/ALOI/32/ folder, appending each of them to
# your list. Each .PNG image should first be loaded into a
# temporary NDArray, just as shown in the Feature
# Representation reading.
#
# Optional: Resample the image down by a factor of two if you
# have a slower computer. You can also convert the image from
# 0-255  to  0.0-1.0  if you'd like, but that will have no
# effect on the algorithm's results.
#

#
# TODO: Convert the list to a dataframe
#
# .. your code here ..  
# .. your code here .. 
AloiDF = pd.DataFrame(samples)
print(AloiDF.shape)


#
# TODO: Implement Isomap here. Reduce the dataframe df down
# to three components, using K=6 for your neighborhood size
#
# .. your code here .
#Running Isomap on first set of images
from sklearn import manifold
iso = manifold.Isomap(n_neighbors = 6, n_components = 3)
iso.fit(AloiDF)

AloiTransform = iso.transform(AloiDF)

#Plotting 2D and 3D images
#
#
# TODO: Create a 2D Scatter plot to graph your manifold. You
# can use either 'o' or '.' as your marker. Graph the first two
# isomap components
#
# .. your code here .. 
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.set_title('2D DataFrame of Isomap on Data')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.scatter(AloiTransform[:, 0], AloiTransform[:, 1], c = colors, marker = 'o')

#
# TODO: Create a 3D Scatter plot to graph your manifold. You
# can use either 'o' or '.' as your marker:
#
# .. your code here .. 

#3D image
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection = '3d')
ax2.set_title('3D DataFrame of Isomap on Data')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.scatter(AloiTransform[:, 0], AloiTransform[:, 1], AloiTransform[:, 2], c = colors, marker = 'o')
# TODO: Once you're done answering the first three questions,
# right before you converted your list to a dataframe, add in
# additional code which also appends to your list the images
# in the Module4/Datasets/ALOI/32_i directory. Re-run your
# assignment and answer the final question below.
#
# .. your code here .. 


plt.show()
