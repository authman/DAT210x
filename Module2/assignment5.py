# %load assignment5.py
import pandas as pd
import numpy as np


#
# TODO:
# Load up the dataset, setting correct header labels.
#
# .. your code here ..
censusData = pd.read_csv("Datasets/census.data", names= ['education', 'age', 'capitalGain', 'race', 'capitalLoss',\
                                                         'hoursPerWeek', 'sex', 'classification'], na_values = '?')
censusData.head()



#
# TODO:
# Use basic pandas commands to look through the dataset... get a
# feel for it before proceeding! Do the data-types of each column
# reflect the values you see when you look through the data using
# a text editor / spread sheet program? If you see 'object' where
# you expect to see 'int32' / 'float64', that is a good indicator
# that there is probably a string or missing value in a column.

#Do dtypes
print(censusData.isnull().values.any(), '\n', '--------------', '\n', censusData.dtypes)

#This shows that we don't have any null value in the dataset but the data contains some unneccasary values cos capitalGain
#    is meant to be an integer
# use `your_data_frame['your_column'].unique()` to see the unique
# values of each column and identify the rogue values. If these
# should be represented as nans, you can convert them using
# na_values when loading the dataframe.
#
# .. your code here ..
censusData.education.unique()
censusData.age.unique()
censusData.capitalGain.unique()
censusData.race.unique()
censusData.capitalLoss.unique()
censusData.hoursPerWeek.unique()
censusData.sex.unique()
censusData.classification.unique()
#
# TODO:
# Look through your data and identify any potential categorical
# features. Ensure you properly encode any ordinal and nominal
# types using the methods discussed in the chapter.
#
# Be careful! Some features can be represented as either categorical
# or continuous (numerical). Think to yourself, does it generally
# make more sense to have a numeric type or a series of categories
# for these somewhat ambigious features?
#
# .. your code here ..

#Code to prepare the ordering of column 'education'
education = censusData.education.unique()
education.tolist()
#To Do:
#Reorder the education list
orderEdu = [7, 3, 0, 5, 1, 12, 2, 9, 6, 8, 10, 11]
neweducation = [education[i] for i in orderEdu]
print(neweducation)
print(education)

#Merge these ordering to education to create a nominal row
censusData.education = censusData.education.astype('category', ordered = True, \
                                                  categories = neweducation)

#censusData.education = censusData.education.dropna()
censusData.capitalGain = censusData.capitalGain.interpolate()
censusData.head()
print('The data type of column education is:', '%s' % censusData.education.dtype) 


#check if any value has Nan value
print(censusData.isnull().any())
censusData.dropna(inplace = True)
#Reprint check if Nan value is in data
censusData.reset_index(drop = True)
#censusData.head()
#You can print 'censusData.isnull().any()' to check if there is still null values
#censusData.education = censusData.education.astype('category', ordered = True, categories = neweducation)
#censusData.education.cat.codes

#Apply get_dummies on 'sex', 'race' and 'classification'
censusData = pd.get_dummies(censusData, columns= ['race', 'sex', 'classification'])

# TODO:
# Print out your dataframe
#
# .. your code here ..
print(censusData.head())

#I have been told that 'classification' should take ordinal values rather than nominal values(questionable though)