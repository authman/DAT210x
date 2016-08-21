import random, math
import pandas as pd
import numpy as np
import scipy.io

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# If you'd like to try this lab with PCA instead of Isomap,
# as the dimensionality reduction technique:
Test_PCA = False

matplotlib.style.use('ggplot') # Look Pretty


def Plot2DBoundary(DTrain, LTrain, DTest, LTest):
  # The dots are training samples (img not drawn), and the pics are testing samples (images drawn)
  # Play around with the K values. This is very controlled dataset so it should be able to get perfect classification on testing entries
  # Play with the K for isomap, play with the K for neighbors. 

  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.set_title('Transformed Boundary, Image Space -> 2D')

  padding = 0.1   # Zoom out
  resolution = 1  # Don't get too detailed; smaller values (finer rez) will take longer to compute
  colors = ['blue','green','orange','red']
  

  # ------

  # Calculate the boundaries of the mesh grid. The mesh grid is
  # a standard grid (think graph paper), where each point will be
  # sent to the classifier (KNeighbors) to predict what class it
  # belongs to. This is why KNeighbors has to be trained against
  # 2D data, so we can produce this countour. Once we have the 
  # label for each point on the grid, we can color it appropriately
  # and plot it.
  x_min, x_max = DTrain[:, 0].min(), DTrain[:, 0].max()
  y_min, y_max = DTrain[:, 1].min(), DTrain[:, 1].max()
  x_range = x_max - x_min
  y_range = y_max - y_min
  x_min -= x_range * padding
  y_min -= y_range * padding
  x_max += x_range * padding
  y_max += y_range * padding

  # Using the boundaries, actually make the 2D Grid Matrix:
  xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                       np.arange(y_min, y_max, resolution))

  # What class does the classifier say about each spot on the chart?
  # The values stored in the matrix are the predictions of the model
  # at said location:
  Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)

  # Plot the mesh grid as a filled contour plot:
  plt.contourf(xx, yy, Z, cmap=plt.cm.terrain, z=-100)


  # ------

  # When plotting the testing images, used to validate if the algorithm
  # is functioning correctly, size them as 5% of the overall chart size
  x_size = x_range * 0.05
  y_size = y_range * 0.05
  
  # First, plot the images in your TEST dataset
  img_num = 0
  for index in LTest.index:
    # DTest is a regular NDArray, so you'll iterate over that 1 at a time.
    x0, y0 = DTest[img_num,0]-x_size/2., DTest[img_num,1]-y_size/2.
    x1, y1 = DTest[img_num,0]+x_size/2., DTest[img_num,1]+y_size/2.

    # DTest = our images isomap-transformed into 2D. But we still want
    # to plot the original image, so we look to the original, untouched
    # dataset (at index) to get the pixels:
    img = df.iloc[index,:].reshape(num_pixels, num_pixels)
    ax.imshow(img, aspect='auto', cmap=plt.cm.gray, interpolation='nearest', zorder=100000, extent=(x0, x1, y0, y1), alpha=0.8)
    img_num += 1


  # Plot your TRAINING points as well... as points rather than as images
  for label in range(len(np.unique(LTrain))):
    indices = np.where(LTrain == label)
    ax.scatter(DTrain[indices, 0], DTrain[indices, 1], c=colors[label], alpha=0.8, marker='o')

  # Plot
  plt.show()  



#
# TODO: Use the same code from Module4/assignment4.py to load up the
# face_data.mat in a dataset called "df". Be sure to calculate the
# num_pixels value, and to rotate the images to being right-side-up
# instead of sideways. This was demonstrated in the M4/A4 code:
#
# .. your code here ..


#
# TODO: Load up your face_labels dataset. It only has a single column, and
# you're only interested in that single column. You will have to slice the 
# column out so that you have access to it as a "Series" rather than as a
# "Dataframe". Use an appropriate indexer to take care of that. Also print
# out the labels and compare to the face_labels.csv file to ensure you
# loaded it correctly
#
# .. your code here ..


#
# TODO: Do train_test_split. Use the same code as on the EdX platform in the
# reading material, but set the random_state=7 for reproduceability, and play
# around with the test_size from 0.10 - 0.20 (10-20%). Your labels are actually
# passed in as a series (instead of as an NDArray) so that you can access
# their underlying indices later on. This is necessary so you can find your samples
# in the original dataframe, which you will use to plot your testing data as images
# rather than as points:
#
# .. your code here ..



if Test_PCA:
  # INFO: PCA is used *before* KNeighbors to simplify your high dimensionality
  # image samples down to just 2 principal components! A lot of information
  # (variance) is lost during the process, as I'm sure you can imagine. But
  # you have to drop the dimension down to two, otherwise you wouldn't be able
  # to visualize a 2D decision surface / boundary. In the wild, you'd probably
  # leave in a lot more dimensions, but wouldn't need to plot the boundary;
  # simply checking the results would suffice.
  #
  # Your model should only be trained (fit) against the training data (data_train)
  # Once you've done this, you need use the model to transform both data_train
  # and data_test from their original high-D image feature space, down to 2D

  #
  #
  # TODO: Implement PCA here. ONLY train against your training data, but
  # transform both your training + test data, storing the results back into
  # data_train, and data_test.
  #
  # .. your code here ..

else:
  # INFO: Isomap is used *before* KNeighbors to simplify your high dimensionality
  # image samples down to just 2 components! A lot of information has been is
  # lost during the process, as I'm sure you can imagine. But if you have
  # non-linear data that can be represented on a 2D manifold, you probably will
  # be left with a far superior dataset to use for classification. Plus by
  # having the images in 2D space, you can plot them as well as visualize a 2D
  # decision surface / boundary. In the wild, you'd probably leave in a lot
  # more dimensions, but wouldn't need to plot the boundary; simply checking
  # the results would suffice.
  #
  # Your model should only be trained (fit) against the training data (data_train)
  # Once you've done this, you need use the model to transform both data_train
  # and data_test from their original high-D image feature space, down to 2D

  #
  # TODO: Implement Isomap here. ONLY train against your training data, but
  # transform both your training + test data, storing the results back into
  # data_train, and data_test.
  #
  # .. your code here ..




#
# TODO: Implement KNeighborsClassifier here. You can use any K value from 1
# through 20, so play around with it and attempt to get good accuracy.
# This is the heart of this assignment: Looking at the 2D points that
# represent your images, along with a list of "answers" or correct class
# labels that those 2d representations should be.
#
# .. your code here ..

# NOTE: K-NEIGHBORS DOES NOT CARE WHAT THE ANSWERS SHOULD BE! In fact, it
# just tosses that information away. All KNeighbors cares about storing is
# your training data (data_train) so that later on when you attempt to
# predict or score samples, it can derive a class for them based on the
# labeling of the sample's near neighbors.


#
# TODO: Calculate + Print the accuracy of the testing set (data_test and
# label_test).
#
# .. your code here ..



# Chart the combined decision boundary, the training data as 2D plots, and
# the testing data as small images so we can visually validate performance.
Plot2DBoundary(data_train, label_train, data_test, label_test)


