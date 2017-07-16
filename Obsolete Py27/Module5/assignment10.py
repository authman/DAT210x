import numpy as np
import pandas as pd

import scipy.io.wavfile as wavfile

# Good Luck!

#
# INFO:
# Samples = Observations. Each audio file will is a single sample
#           in our dataset.
#
# Audio Samples = https://en.wikipedia.org/wiki/Sampling_(signal_processing)
# Each .wav file is actually just a bunch of numeric samples, "sampled"
# from the analog signal. Sampling is a type of discretization. When we
# mention 'samples', we mean observations. When we mention 'audio samples',
# we mean the actually "features" of the audio file.
#
#
# The goal of this lab is to use multi-target, linear regression to generate
# by extrapolation, the missing portion of the test audio file.
#
# Each one audio_sample features will be the output of an equation,
# which is a function of the provided portion of the audio_samples:
#
#    missing_samples = f(provided_samples)
#
# You can experiment with how much of the audio you want to chop off
# and have the computer generate using the Provided_Portion parameter.




#
# TODO: Play with this. This is how much of the audio file will
# be provided, in percent. The remaining percent of the file will
# be generated via linear extrapolation.
Provided_Portion = 0.25



# INFO: You have to download the dataset (audio files) from the website:
# https://github.com/Jakobovski/free-spoken-digit-dataset



#
# TODO: Create a regular ol' Python List called `zero`
#
# .. your code here ..



#
# TODO: Loop through the dataset and load up all 50 of the 0_jackson*.wav
# files using the wavfile.read() method: https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.io.wavfile.read.html
# Be careful! .read() returns a tuple and you're only interested in the audio
# data, and not sample_rate at this point. Inside your for loop, simply
# append the loaded audio data into your Python list `zero`:
#
# .. your code here ..



# 
# TODO: Just for a second, convert zero into a DataFrame. When you do
# so, set the dtype to np.int16, since the input audio files are 16
# bits per sample. If you don't know how to do this, read up on the docs
# here:
# http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html
#
# Since these audio clips are unfortunately not length-normalized,
# we're going to have to just hard chop them to all be the same length.
# Since Pandas would have inserted NANs at any spot to make zero a 
# perfectly rectangular [n_observed_samples, n_audio_samples] array,
# do a dropna on the Y axis here. Then, convert one back into an
# NDArray using yourarrayname.values
#
# .. your code here ..


#
# TODO: It's important to know how (many audio_samples samples) long the
# data is now. 'zero' is currently shaped [n_samples, n_audio_samples],
# so get the n_audio_samples count and store it in a variable called
# n_audio_samples
#
# .. your code here ..



#
# TODO: Create your linear regression model here and store it in a
# variable called 'model'. Don't actually train or do anything else
# with it yet:
#
# .. your code here ..



#
# INFO: There are 50 takes of each clip. You want to pull out just one
# of them, randomly, and that one will NOT be used in the training of
# your model. In other words, the one file we'll be testing / scoring
# on will be an unseen sample, independent to the rest of your
# training set:
from sklearn.utils.validation import check_random_state
rng   = check_random_state(7)  # Leave this alone until you've submitted your lab
random_idx = rng.randint(zero.shape[0])
test  = zero[random_idx]
train = np.delete(zero, [random_idx], axis=0)



# 
# TODO: Print out the shape of train, and the shape of test
# train will be shaped: [n_samples, n_audio_samples], where
# n_audio_samples are the 'features' of the audio file
# train will be shaped [n_audio_features], since it is a single
# sample (audio file, e.g. observation).
#
# .. your code here ..



#
# INFO: The test data will have two parts, X_test and y_test. X_test is
# going to be the first portion of the test audio file, which we will
# be providing the computer as input. y_test, the "label" if you will,
# is going to be the remaining portion of the audio file. Like such, 
# the computer will use linear regression to derive the missing
# portion of the sound file based off of the training data its received!



#
# Save the original 'test' clip, the one you're about to delete
# half of, so that you can compare it to the 'patched' clip once 
# you've generated it. HINT: you should have got the sample_rate
# when you were loading up the .wav files:
wavfile.write('Original Test Clip.wav', sample_rate, test)



#
# TODO: Prepare the TEST date by creating a slice called X_test. It
# should have Provided_Portion * n_audio_samples audio sample features,
# taken from your test audio file, currently stored in the variable
# 'test'. In other words, grab the FIRST Provided_Portion *
# n_audio_samples audio features from test and store it in X_test. This
# should be accomplished using indexing.
#
# .. your code here ..


#
# TODO: If the first Provided_Portion * n_audio_samples features were
# stored in X_test, then we need to also grab the *remaining* audio
# features and store it in y_test. With the remaining features stored
# in there, we will be able to R^2 "score" how well our algorithm did
# in completing the sound file.
#
# .. your code here ..




# 
# TODO: Duplicate the same process for X_train, y_train. The only
# differences being: 1) You will be getting your audio data from
# 'train' instead of from 'test', 2) Remember the shape of train that
# you printed out earlier? You want to do this slicing but for ALL
# samples (observations). For each observation, you want to slice
# the first Provided_Portion * n_audio_samples audio features into
# X_train, and the remaining go into y_tran. All of this should be
# accomplishable using regular indexing in two lines of code.
#
# .. your code here ..



# 
# TODO: SciKit-Learn gets mad if you don't supply your training
# data in the form of a 2D arrays: [n_samples, n_features].
#
# So if you only have one SAMPLE, such as is our case with X_test, 
# and y_test, then by calling .reshape(1, -1), you can turn
# [n_features] into [1, n_features].
#
# On the other hand, if you only have one FEATURE, which currently
# doesn't apply, you can call .reshape(-1, 1) on your data to turn
# [n_samples] into [n_samples, 1]:
#
# .. your code here ..


#
# TODO: Fit your model using your training data and label:
#
# .. your code here ..


# 
# TODO: Use your model to predict the 'label' of X_test. Store the
# resulting prediction in a variable called y_test_prediction
#
# .. your code here ..


# INFO: SciKit-Learn will use float64 to generate your predictions
# so let's take those values back to int16:
y_test_prediction = y_test_prediction.astype(dtype=np.int16)



# 
# TODO: Score how well your prediction would do for some good laughs,
# by passing in your test data and test label (y_test).
#
# .. your code here ..
print "Extrapolation R^2 Score: ", score


#
# First, take the first Provided_Portion portion of the test clip, the
# part you fed into your linear regression model. Then, stitch that
# together with the abomination the predictor model generated for you,
# and then save the completed audio clip:
completed_clip = np.hstack((X_test, y_test_prediction))
wavfile.write('Extrapolated Clip.wav', sample_rate, completed_clip[0])



#
# INFO: Congrats on making it to the end of this crazy lab =) !
#
