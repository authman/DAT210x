import pandas as pd

import matplotlib.pyplot as plt
import matplotlib

from sklearn.cluster import KMeans

matplotlib.style.use('ggplot') # Look Pretty

def showandtell(title=None):
  if title != None: plt.savefig(title + ".png", bbox_inches='tight', dpi=300)
  plt.show()
  
# INFO: This dataset has call records for 10 users tracked over the course of 3 years.
# Your job is to find out where the users likely live and work at!

df = pd.read_csv('Datasets/CDR.csv')
df = df.dropna(axis=0, how='any')

print df.head(10)
print 'Records number: ' + str(df['In'].count())
print ''

df['CallDate'] = pd.to_datetime(df['CallDate'])
df['CallTime'] = pd.to_timedelta(df['CallTime'])
#print df.dtypes


phones = df['In'].unique().tolist()

for phone in phones:
    print phone
    
    user = df[df['In'] == phone]
    user.plot.scatter(x='TowerLon', y='TowerLat', c='gray', alpha=0.1, title='Call Locations')
    
    user = user[(user['DOW'] == 'Sat') | (user['DOW'] == 'Sun')]
    user = user[(user['CallTime'] < '06:00:00') | (user['CallTime'] > '22:00:00')]
    
    print len(user)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(user.TowerLon, user.TowerLat, c='g', marker='o', alpha=0.2)
    ax.set_title('Weekend Calls (<6am or >10p)')
    
    user = user[['TowerLat', 'TowerLon']]
    kmeans = KMeans(1)
    kmeans.fit(user)
    
    centroids = kmeans.cluster_centers_
    ax.scatter(centroids[:,1], centroids[:,0], marker='x', c='red')
    print centroids

# showandtell()  # TODO: Comment this line out when you're ready to proceed

#
# TODO: Repeat the above steps for all 10 individuals, being sure to record their approximate home
# locations. You might want to use a for-loop, unless you enjoy typing.
#
# .. your code here ..

