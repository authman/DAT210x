# %load assignment3.py
import pandas as pd

# TODO: Load up the dataset
# Ensuring you set the appropriate header column names
#
# .. your code here ..
servoData = pd.read_csv('Datasets/servo.data', names = ['motor', 'screw', 'pgain', 'vgain', 'class'])
print(servoData.head())


# TODO: Create a slice that contains all entries
# having a vgain equal to 5. Then print the 
# length of (# of samples in) that slice:
#
# .. your code here ..
vgainLess = servoData[servoData.vgain == 5]
print(len(vgainLess))

# TODO: Create a slice that contains all entries
# having a motor equal to E and screw equal
# to E. Then print the length of (# of
# samples in) that slice:
#
# .. your code here ..
newSlice =servoData[(servoData.motor == 'E') & (servoData.screw == 'E')] 
print(len(newSlice))


# TODO: Create a slice that contains all entries
# having a pgain equal to 4. Use one of the
# various methods of finding the mean vgain
# value for the samples in that slice. Once
# you've found it, print it:
#
# .. your code here ..
lastSlice = servoData[servoData.pgain == 4].vgain.mean() #Tried out chaining the process
print(lastSlice) 
#lastSlice = servoData[servoData.pgain == 4]
#print(lastSlice.vgain.mean())


# TODO: (Bonus) See what happens when you run
# the .dtypes method on your dataframe!
print(servoData.dtypes)