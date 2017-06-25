import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

# Look pretty...
# matplotlib.style.use('ggplot')
plt.style.use('ggplot')

#
# TODO: Load up the Seeds Dataset into a Dataframe
# It's located at 'Datasets/wheat.data'
# 
df = pd.read_csv('Datasets/wheat.data', index_col=0)

fig = plt.figure()

#
# TODO: Create a new 3D subplot using fig. Then use the
# subplot to graph a 3D scatter plot using the area,
# perimeter and asymmetry features. Be sure to use the
# optional display parameter c='red', and also label your
# axes
# 
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('area')
ax.set_ylabel('perimeter')
ax.set_zlabel('asymmetry')
ax.scatter(df.area, df.perimeter, df.asymmetry, c='red')

fig = plt.figure()
#
# TODO: Create a new 3D subplot using fig. Then use the
# subplot to graph a 3D scatter plot using the width,
# groove and length features. Be sure to use the
# optional display parameter c='green', and also label your
# axes
# 
ax2 = fig.add_subplot(111, projection='3d')
ax2.set_xlabel('width')
ax2.set_ylabel('groove')
ax2.set_zlabel('length')
ax2.scatter(df.width, df.groove, df.length, c='g')

plt.show()


