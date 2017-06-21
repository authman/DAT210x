import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib

from sklearn.cluster import KMeans

matplotlib.style.use('ggplot') # Look Pretty

#
# INFO: This dataset has call records for 10 users tracked over the course of 3 years.
# Your job is to find out where the users likely live at!


def showandtell(title=None):
  if title != None: plt.savefig(title + ".png", bbox_inches='tight', dpi=300)
  plt.show()
  # exit()

def clusterInfo(model):
  print "Cluster Analysis Inertia: ", model.inertia_
  print '------------------------------------------'
  for i in range(len(model.cluster_centers_)):
    print "\n  Cluster ", i
    print "    Centroid ", model.cluster_centers_[i]
    print "    #Samples ", (model.labels_==i).sum() # NumPy Power

# Find the cluster with the least # attached nodes
def clusterWithFewestSamples(model):
  # Ensure there's at least on cluster...
  minSamples = len(model.labels_)
  minCluster = 0
  for i in range(len(model.cluster_centers_)):
    if minSamples > (model.labels_==i).sum():
      minCluster = i
      minSamples = (model.labels_==i).sum()
  print "\n  Cluster With Fewest Samples: ", minCluster
  return (model.labels_==minCluster)


def doKMeans(data, ax, clusters=0):
  coords = data[['TowerLat', 'TowerLon']]
  model = KMeans(clusters)
  model.fit(coords)
  
  centroids = model.cluster_centers_
  
  ax.scatter(centroids[:,1], centroids[:,0], marker='x', color='r')
  return model

df = pd.read_csv('Datasets/CDR.csv')

df['CallDate'] = pd.to_datetime(df['CallDate'])
df['CallTime'] = pd.to_timedelta(df['CallTime'])

print df.head()
print df.dtypes

unique_numbers = df['In'].unique()


for phone in unique_numbers:
  print "\n\nExamining person with number: ", phone
  user = df[df['In'] == phone]
  
  user = user[(user['DOW'] != 'Sat') & (user['DOW'] != 'Sun')]

  user = user[user['CallTime'] < '17:00:00']

  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.scatter(user.TowerLon, user.TowerLat, c='g', marker='.', alpha=0.5)
  ax.set_title('Weekday calls (<5pm)')

  model = doKMeans(user, ax, 4)
  #midWayClusterIndices = clusterWithFewestSamples(model)
  #midWaySamples = user[midWayClusterIndices]
  #print "    Its Waypoint Time: ", midWaySamples.CallTime.mean()

  clusterInfo(model)

  # Let's visualize the results!
  # First draw the X's for the clusters:
  ax.scatter(model.cluster_centers_[:,1], model.cluster_centers_[:,0], s=169, c='r', marker='x', alpha=0.8, linewidths=2)

  # Then save the results:
  showandtell('Weekday Calls Centroids')  # Comment this line out when you're ready to proceed
