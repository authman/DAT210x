import matplotlib.pyplot as plt
import pandas as pd

# Look Pretty
plt.style.use('ggplot')

def doKMeans(df):
  #
  # INFO: Plot your data with a '.' marker, with 0.3 alpha at the Longitude,
  # and Latitude locations in your dataset. Longitude = x, Latitude = y
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.scatter(df.Longitude, df.Latitude, marker='.', alpha=0.3)

  #
  # TODO: Filter df so that you're only looking at Longitude and Latitude,
  # since the remaining columns aren't really applicable for this purpose.
  #

  #
  # TODO: Use K-Means to try and find seven cluster centers in this df.
  # Be sure to name your kmeans model `model` so that the printing works.
  #
  # .. your code here ..

  #
  # INFO: Print and plot the centroids...
  centroids = model.cluster_centers_
  ax.scatter(centroids[:,0], centroids[:,1], marker='x', c='red', alpha=0.5, linewidths=3, s=169)
  print centroids



#
# TODO: Load your dataset after importing Pandas
#
df = pd.read_csv('Datasets/Crimes.csv', index_col=0)

#
# TODO: Drop any ROWs with nans in them
#
# .. your code here ..


#
# TODO: Print out the dtypes of your dset
#
# .. your code here ..


#
# Coerce the 'Date' feature (which is currently a string object) into real date,
# and confirm by re-printing the dtypes. NOTE: This is a slow process...
#
# .. your code here ..


# INFO: Print & Plot your data
doKMeans(df)


#
# TODO: Filter out the data so that it only contains samples that have
# a Date > '2011-01-01', using indexing. Then, in a new figure, plot the
# crime incidents, as well as a new K-Means run's centroids.
#
# .. your code here ..



# INFO: Print & Plot your data
doKMeans(df)
plt.show()


