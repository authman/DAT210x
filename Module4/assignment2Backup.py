import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import assignment2_helper as helper
%matplotlib inline

# Look pretty...
matplotlib.style.use('ggplot')


# Do * NOT * alter this line, until instructed!
scaleFeatures = False


# TODO: Load up the dataset and remove any and all
# Rows that have a nan. You should be a pro at this
# by now ;-)
#
# .. your code here ..
kidneyData = pd.read_csv('Datasets/kidney_disease.csv')
kidneyDataNull = kidneyData.dropna(axis = 0, how='any' ).reset_index(drop = True)

# Create some color coded labels; the actual label feature
# will be removed prior to executing PCA, since it's unsupervised.
# You're only labeling by color so you can see the effects of PCA
labels = ['red' if i=='ckd' else 'green' for i in kidneyDataNull.classification]


# TODO: Use an indexer to select only the following columns:
#       ['bgr','wc','rc']
#
# .. your code here ..

selectMain = kidneyDataNull.loc[:, ['bgr', 'wc', 'rc']]

# TODO: Print out and check your dataframe's dtypes. You'll probably
# want to call 'exit()' after you print it out so you can stop the
# program's execution.
#
print(selectMain.dtypes)


# You can either take a look at the dataset webpage in the attribute info
# section: https://archive.ics.uci.edu/ml/datasets/Chronic_Kidney_Disease
# or you can actually peek through the dataframe by printing a few rows.
# What kind of data type should these three columns be? If Pandas didn't
# properly detect and convert them to that data type for you, then use
# an appropriate command to coerce these features into the right type.
#
# .. your code here ..

selectMain = selectMain.apply(lambda s: pd.to_numeric(s))
print(selectMain.dtypes)
# TODO: PCA Operates based on variance. The variable with the greatest
# variance will dominate. Go ahead and peek into your data using a
# command that will check the variance of every feature in your dataset.

print(selectMain.var())
# Print out the results. Also print out the results of running .describe
# on your dataset.
#
# Hint: If you don't see all three variables: 'bgr','wc' and 'rc', then
# you probably didn't complete the previous step properly.
#
# .. your code here ..
print(selectMain.describe())


# TODO: This method assumes your dataframe is called df. If it isn't,
# make the appropriate changes. Don't alter the code in scaleFeatures()
# just yet though!
#
# .. your code adjustment here ..
if scaleFeatures: selectMain = helper.scaleFeatures(selectMain)


# TODO: Run PCA on your dataset and reduce it to 2 components
# Ensure your PCA instance is saved in a variable called 'pca',
# and that the results of your transformation are saved in 'T'.
#
# .. your code here ..
    #import PCA from sklearn
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
pca.fit(selectMain)
T = pca.transform(selectMain)


# Plot the transformed data as a scatter plot. Recall that transforming
# the data will result in a NumPy NDArray. You can either use MatPlotLib
# to graph it directly, or you can convert it to DataFrame and have pandas
# do it for you.
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.set_title('Unreduced Format of selectMain')
ax.set_xlabel('bgr')
ax.set_ylabel('rc')
ax.set_zlabel('wc')
ax.scatter(selectMain.bgr, selectMain.rc, selectMain.wc, c = 'green', marker = '.')
#

#plot reduced form of selectMain
fig = plt.figure()
ax2 = fig.add_subplot(111)
ax2.set_title('PCA reduction of Selected Features of KidneyData')
ax2.set_xlabel('Component 1')
ax2.set_ylabel('Component 2')
ax2.scatter(T[:, 0], T[:, 1], c = 'orange' , marker = 'o', alpha = 0.75)
plt.show()
# Since we've already demonstrated how to plot directly with MatPlotLib in
# Module4/assignment1.py, this time we'll convert to a Pandas Dataframe.
#
# Since we transformed via PCA, we no longer have column names. We know we
# are in P.C. space, so we'll just define the coordinates accordingly:
ax = helper.drawVectors(T, pca.components_, selectMain.columns.values, plt, scaleFeatures)
T = pd.DataFrame(T)
T.columns = ['component1', 'component2']

T.plot.scatter(x='component1', y='component2', marker='o', c=labels, alpha=0.75, ax=ax)
plt.show()

