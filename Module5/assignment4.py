import numpy as np
import pandas as pd
from sklearn import preprocessing

import matplotlib.pyplot as plt
import matplotlib


#
# TODO: Parameters to play around with
PLOT_TYPE_TEXT = False    # If you'd like to see indices
PLOT_VECTORS = True       # If you'd like to see your original features in P.C.-Space


matplotlib.style.use('ggplot') # Look Pretty
c = ['red', 'green', 'blue', 'orange', 'yellow', 'brown']

def drawVectors(transformed_features, components_, columns, plt):
  num_columns = len(columns)

  # This function will project your *original* feature (columns)
  # onto your principal component feature-space, so that you can
  # visualize how "important" each one was in the
  # multi-dimensional scaling
  
  # Scale the principal components by the max value in
  # the transformed set belonging to that component
  xvector = components_[0] * max(transformed_features[:,0])
  yvector = components_[1] * max(transformed_features[:,1])

  ## Visualize projections

  # Sort each column by its length. These are your *original*
  # columns, not the principal components.
  import math
  important_features = { columns[i] : math.sqrt(xvector[i]**2 + yvector[i]**2) for i in range(num_columns) }
  important_features = sorted(zip(important_features.values(), important_features.keys()), reverse=True)
  print "Projected Features by importance:\n", important_features

  ax = plt.axes()

  for i in range(num_columns):
    # Use an arrow to project each original feature as a
    # labeled vector on your principal component axes
    plt.arrow(0, 0, xvector[i], yvector[i], color='b', width=0.0005, head_width=0.02, alpha=0.75, zorder=600000)
    plt.text(xvector[i]*1.2, yvector[i]*1.2, list(columns)[i], color='b', alpha=0.75, zorder=600000)
  return ax
    

def doPCA(data, dimensions=2):
  from sklearn.decomposition import PCA
  model = PCA(n_components=dimensions, svd_solver='randomized', random_state=7)
  model.fit(data)
  return model


def doKMeans(data, clusters=0):
  #
  # TODO: Do the KMeans clustering here, passing in the # of clusters parameter
  # and fit it against your data. Then, return a tuple containing the cluster
  # centers and the labels.
  #
  # Hint: Just like with doPCA above, you will have to create a variable called
  # `model`, which is a SKLearn K-Means model for this to work.
  #
  # .. your code here ..
  return model.cluster_centers_, model.labels_


#
# TODO: Load up the dataset. It may or may not have nans in it. Make
# sure you catch them and destroy them, by setting them to '0'. This is valid
# for this dataset, since if the value is missing, you can assume no $ was spent
# on it.
#
# .. your code here ..

#
# TODO: As instructed, get rid of the 'Channel' and 'Region' columns, since
# you'll be investigating as if this were a single location wholesaler, rather
# than a national / international one. Leaving these fields in here would cause
# KMeans to examine and give weight to them.
#
# .. your code here ..


#
# TODO: Before unitizing / standardizing / normalizing your data in preparation for
# K-Means, it's a good idea to get a quick peek at it. You can do this using the
# .describe() method, or even by using the built-in pandas df.plot.hist()
#
# .. your code here ..


#
# INFO: Having checked out your data, you may have noticed there's a pretty big gap
# between the top customers in each feature category and the rest. Some feature
# scaling algos won't get rid of outliers for you, so it's a good idea to handle that
# manually---particularly if your goal is NOT to determine the top customers. After
# all, you can do that with a simple Pandas .sort_values() and not a machine
# learning clustering algorithm. From a business perspective, you're probably more
# interested in clustering your +/- 2 standard deviation customers, rather than the
# creme dela creme, or bottom of the barrel'ers
#
# Remove top 5 and bottom 5 samples for each column:
drop = {}
for col in df.columns:
  # Bottom 5
  sort = df.sort_values(by=col, ascending=True)
  if len(sort) > 5: sort=sort[:5]
  for index in sort.index: drop[index] = True # Just store the index once

  # Top 5
  sort = df.sort_values(by=col, ascending=False)
  if len(sort) > 5: sort=sort[:5]
  for index in sort.index: drop[index] = True # Just store the index once

#
# INFO Drop rows by index. We do this all at once in case there is a
# collision. This way, we don't end up dropping more rows than we have
# to, if there is a single row that satisfies the drop for multiple columns.
# Since there are 6 rows, if we end up dropping < 5*6*2 = 60 rows, that means
# there indeed were collisions.
print "Dropping {0} Outliers...".format(len(drop))
df.drop(inplace=True, labels=drop.keys(), axis=0)
print df.describe()


#
# INFO: What are you interested in?
#
# Depending on what you're interested in, you might take a different approach
# to normalizing/standardizing your data.
# 
# You should note that all columns left in the dataset are of the same unit.
# You might ask yourself, do I even need to normalize / standardize the data?
# The answer depends on what you're trying to accomplish. For instance, although
# all the units are the same (generic money unit), the price per item in your
# store isn't. There may be some cheap items and some expensive one. If your goal
# is to find out what items people tend to buy together but you didn't 
# "unitize" properly before running kMeans, the contribution of the lesser priced
# item would be dwarfed by the more expensive item. This is an issue of scale.
#
# For a great overview on a few of the normalization methods supported in SKLearn,
# please check out: https://stackoverflow.com/questions/30918781/right-function-for-normalizing-input-of-sklearn-svm
#
# Suffice to say, at the end of the day, you're going to have to know what question
# you want answered and what data you have available in order to select the best
# method for your purpose. Luckily, SKLearn's interfaces are easy to switch out
# so in the mean time, you can experiment with all of them and see how they alter
# your results.
#
#
# 5-sec summary before you dive deeper online:
#
# NORMALIZATION: Let's say your user spend a LOT. Normalization divides each item by
#                the average overall amount of spending. Stated differently, your
#                new feature is = the contribution of overall spending going into
#                that particular item: $spent on feature / $overall spent by sample
#
# MINMAX:        What % in the overall range of $spent by all users on THIS particular
#                feature is the current sample's feature at? When you're dealing with
#                all the same units, this will produce a near face-value amount. Be
#                careful though: if you have even a single outlier, it can cause all
#                your data to get squashed up in lower percentages.
#                Imagine your buyers usually spend $100 on wholesale milk, but today
#                only spent $20. This is the relationship you're trying to capture 
#                with MinMax. NOTE: MinMax doesn't standardize (std. dev.); it only
#                normalizes / unitizes your feature, in the mathematical sense.
#                MinMax can be used as an alternative to zero mean, unit variance scaling.
#                [(sampleFeatureValue-min) / (max-min)] * (max-min) + min
#                Where min and max are for the overall feature values for all samples.


#
# TODO: Un-comment just ***ONE*** of lines at a time and see how alters your results
# Pay attention to the direction of the arrows, as well as their LENGTHS
#T = preprocessing.StandardScaler().fit_transform(df)
#T = preprocessing.MinMaxScaler().fit_transform(df)
#T = preprocessing.MaxAbsScaler().fit_transform(df)
#T = preprocessing.Normalizer().fit_transform(df)
T = df # No Change


#
# INFO: Sometimes people perform PCA before doing KMeans, so that KMeans only
# operates on the most meaningful features. In our case, there are so few features
# that doing PCA ahead of time isn't really necessary, and you can do KMeans in
# feature space. But keep in mind you have the option to transform your data to
# bring down its dimensionality. If you take that route, then your Clusters will
# already be in PCA-transformed feature space, and you won't have to project them
# again for visualization.


# Do KMeans
n_clusters = 3
centroids, labels = doKMeans(T, n_clusters)


#
# TODO: Print out your centroids. They're currently in feature-space, which
# is good. Print them out before you transform them into PCA space for viewing
#
# .. your code here ..


# Do PCA *after* to visualize the results. Project the centroids as well as 
# the samples into the new 2D feature space for visualization purposes.
display_pca = doPCA(T)
T = display_pca.transform(T)
CC = display_pca.transform(centroids)


# Visualize all the samples. Give them the color of their cluster label
fig = plt.figure()
ax = fig.add_subplot(111)
if PLOT_TYPE_TEXT:
  # Plot the index of the sample, so you can further investigate it in your dset
  for i in range(len(T)): ax.text(T[i,0], T[i,1], df.index[i], color=c[labels[i]], alpha=0.75, zorder=600000)
  ax.set_xlim(min(T[:,0])*1.2, max(T[:,0])*1.2)
  ax.set_ylim(min(T[:,1])*1.2, max(T[:,1])*1.2)
else:
  # Plot a regular scatter plot
  sample_colors = [ c[labels[i]] for i in range(len(T)) ]
  ax.scatter(T[:, 0], T[:, 1], c=sample_colors, marker='o', alpha=0.2)


# Plot the Centroids as X's, and label them
ax.scatter(CC[:, 0], CC[:, 1], marker='x', s=169, linewidths=3, zorder=1000, c=c)
for i in range(len(centroids)): ax.text(CC[i, 0], CC[i, 1], str(i), zorder=500010, fontsize=18, color=c[i])


# Display feature vectors for investigation:
if PLOT_VECTORS: drawVectors(T, display_pca.components_, df.columns, plt)


# Add the cluster label back into the dataframe and display it:
df['label'] = pd.Series(labels, index=df.index)
print df

plt.show()
