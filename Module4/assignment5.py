import os
import pandas as pd

from scipy import misc
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt

from sklearn import manifold

plt.style.use('ggplot')

# TODO: Start by creating a regular old, plain, "vanilla"
# python list. You can call it 'samples'.
samples = []
colors = []
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
def append_images(samples, path, colors, color):     
    for root, dirs, files in os.walk(path):
        for fname in files:
            img = misc.imread(path + fname)
            samples.append(img[::2,::2].reshape(-1))
            colors.append(color)

append_images(samples, 'Datasets/ALOI/32/', colors, 'b')

# TODO: Once you're done answering the first three questions,
# right before you converted your list to a dataframe, add in
# additional code which also appends to your list the images
# in the Module4/Datasets/ALOI/32_i directory. Re-run your
# assignment and answer the final question below.
#
append_images(samples, 'Datasets/ALOI/32i/', colors, 'r')

df = pd.DataFrame(samples)

# TODO: Implement Isomap here. Reduce the dataframe df down
# to three components, using K=6 for your neighborhood size
iso = manifold.Isomap(n_neighbors=6, n_components=3)
iso.fit(df)
T = iso.transform(df)

#
# TODO: Create a 2D Scatter plot to graph your manifold. You
# can use either 'o' or '.' as your marker. Graph the first two
# isomap components
#
def plot_2D(T, colors, x=0, y=1):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('ISO 2D scatter plot')
    ax.set_xlabel('C: {0}'.format(x))
    ax.set_ylabel('C: {0}'.format(y))
    ax.scatter(T[:,x], T[:,y], marker='o', alpha=1.0, c=colors)

#
# TODO: Create a 3D Scatter plot to graph your manifold. You
# can use either 'o' or '.' as your marker:
#
def plot_3D(T, colors, x=0, y=1, z=2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('ISO 3D scatter plot')
    ax.set_xlabel('C: {0}'.format(x))
    ax.set_ylabel('C: {0}'.format(y))
    ax.set_zlabel('C: {0}'.format(z))
    ax.scatter(T[:,x], T[:,y], T[:,z], marker='o', alpha=1.0, c=colors)


plot_2D(T, colors)
plot_3D(T, colors)
plt.show()

