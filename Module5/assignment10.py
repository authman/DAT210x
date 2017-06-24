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

import scipy.io.wavfile as wv

# INFO: You have to download the dataset (audio files) from the website:
# https://github.com/Jakobovski/free-spoken-digit-dataset

zero = []

for i in range(0, 50):
  sample_rate, wv_data = wv.read('Datasets/audio/0_jackson_' + str(i) + '.wav')
  zero.append(wv_data)

zero = pd.DataFrame(zero, dtype=np.int16)
zero = zero.dropna(axis=1)
zero = zero.values

n_samples, n_audio_samples = zero.shape

from sklearn import linear_model
model = linear_model.LinearRegression()

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

print train.shape
print test.shape

# INFO: The test data will have two parts, X_test and y_test. X_test is
# going to be the first portion of the test audio file, which we will
# be providing the computer as input. y_test, the "label" if you will,
# is going to be the remaining portion of the audio file. Like such, 
# the computer will use linear regression to derive the missing
# portion of the sound file based off of the training data its received!

original_test = test.copy()

# HINT: you should have got the sample_rate
# when you were loading up the .wav files:
wavfile.write('Original Test Clip.wav', sample_rate, test)

boundry = Provided_Portion * n_audio_samples

X_test = test[:boundry].reshape(1, -1)
Y_test = test[boundry:].reshape(1, -1)

X_train = train[:,:boundry]
Y_train = train[:,boundry:]

print X_test.shape
print Y_test.shape
print X_train.shape
print Y_train.shape

model.fit(X_train, Y_train)

y_test_prediction = model.predict(X_test)

# INFO: SciKit-Learn will use float64 to generate your predictions
# so let's take those values back to int16:
y_test_prediction = y_test_prediction.astype(dtype=np.int16)
print y_test_prediction

score = model.score(X_test, Y_test)
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
