# %load assignment2.py #automatically load the python notebook into ipython notebook
# %%writefile [-a] assignment2.py #fill the cells back into the python notebook
#%save assignment2.py
import pandas as pd

# TODO: Load up the 'tutorial.csv' dataset

# .. your code here ..
data = pd.read_csv("Datasets/tutorial.csv")



# TODO: Print the results of the .describe() method
#
# .. your code here ..
print(data.describe())


# TODO: Figure out which indexing method you need to
# use in order to index your dataframe with: [2:4,'col3']
# And print the results
#
# .. your code here ..
index3 = data.ix[2:4, 'col3']
print(index3)