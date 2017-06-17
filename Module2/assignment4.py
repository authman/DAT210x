import pandas as pd


# TODO: Load up the table, and extract the dataset
# out of it. If you're having issues with this, look
# carefully at the sample code provided in the reading
#
df = pd.read_html('http://www.espn.com/nhl/statistics/player/_/stat/points/sort/points/year/2015/seasontype/2', 
                  header=1)[0]
#print df.head(5)

# TODO: Rename the columns so that they are similar to the
# column definitions provided to you on the website.
# Be careful and don't accidentially use any names twice.
#
df.columns = ['RK', 'PLAYER', 'TEAM', 'GP', 'G', 'A', 'PTS', 'RATIO', 'PIM', 'PTS_G', 'SOG', 'PCT', 'GWG', 'PP_G', 'PP_A', 'SH_G', 'SH_A']
#print df.head(5)
print len(df)

# TODO: Get rid of any row that has at least 4 NANs in it,
# e.g. that do not contain player points statistics
#
df = df.dropna(axis=0, thresh=4)
print len(df)

# TODO: At this point, look through your dataset by printing
# it. There probably still are some erroneous rows in there.
# What indexing command(s) can you use to select all rows
# EXCEPT those rows?
#

df = df[df.PTS != 'PTS']
print len(df)
        
# TODO: Get rid of the 'RK' column
#
df = df.drop(axis=1, labels=['RK'])
#print df.head(5)
#print len(df)

# TODO: Ensure there are no holes in your index by resetting
# it. By the way, don't store the original index
#
df = df.reset_index(drop=True)
#print df.head(10)

# TODO: Check the data type of all columns, and ensure those
# that should be numeric are numeric
#
#print list(df.columns.values)

df.GP = pd.to_numeric(df.GP, errors='coerce')
df.G = pd.to_numeric(df.G, errors='coerce')
df.A = pd.to_numeric(df.A, errors='coerce')
df.PTS = pd.to_numeric(df.PTS, errors='coerce')
df.RATIO = pd.to_numeric(df.RATIO, errors='coerce')
df.PIM = pd.to_numeric(df.PIM, errors='coerce')
df.PTS_G = pd.to_numeric(df.PTS_G, errors='coerce')
df.SOG = pd.to_numeric(df.SOG, errors='coerce')
df.PCT = pd.to_numeric(df.PCT, errors='coerce')
df.GWG = pd.to_numeric(df.GWG, errors='coerce')
df.PP_G = pd.to_numeric(df.PP_G, errors='coerce')
df.PP_A = pd.to_numeric(df.PP_A, errors='coerce')
df.SH_G = pd.to_numeric(df.SH_G, errors='coerce')
df.SH_A = pd.to_numeric(df.SH_A, errors='coerce')

#print df.dtypes
# TODO: Your dataframe is now ready! Use the appropriate 
# commands to answer the questions on the course lab page.
#
# .. your code here ..
#print df
print df.head(50)
print 'Answers:'
print len(df)
print len(df.PCT.unique())
print df.ix[15, 'GP'] + df.ix[16, 'GP']
