import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans

# Look Pretty
plt.style.use('ggplot')

def doKMeans(df):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(df.Longitude, df.Latitude, marker='.', alpha=0.3)
    
    df = df[['Longitude', 'Latitude']]
    model = KMeans(n_clusters=7)
    model.fit(df)
    
    centroids = model.cluster_centers_
    ax.scatter(centroids[:,0], centroids[:,1], marker='x', c='red', alpha=0.5, linewidths=3, s=169)
    print centroids



#
# TODO: Load your dataset after importing Pandas
#
df = pd.read_csv('Datasets/Crimes.csv', index_col=0)
df = df.dropna(axis=0,how='any')

print df.dtypes
df['Date'] = pd.to_datetime(df['Date'])
print df.dtypes

# INFO: Print & Plot your data
doKMeans(df)


#
# TODO: Filter out the data so that it only contains samples that have
# a Date > '2011-01-01', using indexing. Then, in a new figure, plot the
# crime incidents, as well as a new K-Means run's centroids.
#
# .. your code here ..
df = df[df['Date'] > '2011-01-01']


# INFO: Print & Plot your data
doKMeans(df)
plt.show()


