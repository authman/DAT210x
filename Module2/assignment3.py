import pandas as pd

# TODO: Load up the dataset
# Ensuring you set the appropriate header column names
#
names=['motor', 'screw', 'pgain', 'vgain', 'class']
df = pd.read_csv('Datasets/servo.data', names=names)
print df


# TODO: Create a slice that contains all entries
# having a vgain equal to 5. Then print the 
# length of (# of samples in) that slice:
#
df_vgain = df[df.vgain == 5]
print len(df_vgain)


# TODO: Create a slice that contains all entries
# having a motor equal to E and screw equal
# to E. Then print the length of (# of
# samples in) that slice:
#
df_motor_and_screw = df[(df.motor == 'E') & (df.screw == 'E')]
print len(df_motor_and_screw)


# TODO: Create a slice that contains all entries
# having a pgain equal to 4. Use one of the
# various methods of finding the mean vgain
# value for the samples in that slice. Once
# you've found it, print it:
#
df_pgain = df[df.pgain == 4]
print df_pgain.describe().vgain[1]



# TODO: (Bonus) See what happens when you run
# the .dtypes method on your dataframe!
print df.dtypes


