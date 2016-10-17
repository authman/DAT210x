# %load assignment4.py
import pandas as pd

# TODO: Load up the table, and extract the dataset
# out of it. If you're having issues with this, look
# carefully at the sample code provided in the reading
#
# .. your code here ..
#using Beafiul soup to get the table and save it as a csv file
import csv
import urllib
from bs4 import BeautifulSoup

with open('listing.csv', 'w') as f:
    writer = csv.writer(f)
    url = "http://www.espn.com/nhl/statistics/player/_/stat/points/sort/points/year/2015/seasontype/2"
    u = urllib.request.urlopen(url)
    try:
        html = u.read()
    finally:
        u.close()
    soup=BeautifulSoup(html)
    for tr in soup.find_all('tr')[2:]:
        tds = tr.find_all('td')
        row = [elem.text for elem in tds] #initially there was a 'elem.text.encoding('utf-8') here and was giving me issue
        writer.writerow(row)


listing = pd.read_csv("listing.csv", names = ['RK', 'PLAYER', 'TEAM', 'GP', 'G', 'A', 'PTS', '+/-',\
                                              'PIM', 'PTS/G', 'SOG', 'PCT', 'GWG', 'G', 'A', 'G', 'A'],\
                      encoding = 'ISO-8859-1')
#listing.drop('RK', axis = 1, inplace = True)
#listing.iloc[:, 1].map(lambda x: x.rstrip("'").lstrip("'")) a very good tool to keep 'rstrip()' and 'lstrip()' on pd.Series
listing.head()
# TODO: Rename the columns so that they match the
# column definitions provided to you on the website
#
# .. your code here ..


# TODO: Get rid of any row that has at least 4 NANs in it
#
# .. your code here ..

listing_notNull = listing.dropna(axis = 0, thresh= 4)
print(listing_notNull.head())

# TODO: At this point, look through your dataset by printing
# it. There probably still are some erroneous rows in there.
# What indexing command(s) can you use to select all rows
# EXCEPT those rows?
#
# .. your code here ..


#suggestions of more pythonic way to do this part!
listing_notDup = listing_notNull[listing_notNull['RK'] != 'RK']
print(listing_notDup.head())

# TODO: Get rid of the 'RK' column
#
# .. your code here ..

listing_notDup.drop('RK', axis = 1, inplace = True)
# TODO: Ensure there are no holes in your index by resetting
# it. By the way, don't store the original index
#
# .. your code here ..
listing_notDup = listing_notDup.reset_index(drop = True) #'drop = True' ensures that there is no column name index in the data
# TODO: Check the data type of all columns, and ensure those
# that should be numeric are numeric

#Data is all objects because of the way that the data was read in
listing_notDup = listing_notDup.apply(lambda x: pd.to_numeric(x, errors = 'ignore'))
print(listing_notDup.dtypes)

# TODO: Your dataframe is now ready! Use the appropriate 
# commands to answer the questions on the course lab page.

#How many unique rows exits after the cleaning operation
listing_notDup.info()

#How many unique values are in PCT
print(listing_notDup.PCT.nunique()) #nunique() does the job, another lesson for easy analysis the .columnname type of python coding is
#                             good and allows for more expression of possible methods on column series
