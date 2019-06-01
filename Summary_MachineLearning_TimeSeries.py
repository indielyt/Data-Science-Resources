# Datacamp MACHINE LEARNING FOR TIME SERIES IN PYTHON

# Chapter 1: Time Series and Machine Learning Primer
	
    # Plotting a time series (I)
    # Plotting a time series (II)
    # Fitting a simple model: classification
    # Predicting using a classification model
    # Predicting using a regression model
    # Inspecting the classification data
    # Inspecting the regression data

# Chapter 2: Time Series as Inputs to a Model

    # Many repetitions of sounds
    # Invariance in time
    # Build a classification model
    # Calculating the envelope of sound
    # Calculating features from the envelope
    # Derivative features: The tempogram
    # Spectrograms of heartbeat audio
    # Engineering spectral features
    # Combining many features in a classifier

# Chapter 3: Predicting Time Series Data

    # Introducing the dataset
    # Fitting a simple regression model
    # Visualizing predicted values
    # Visualizing messy data
    # Imputing missing values
    # Transforming raw data
    # Handling outliers
    # Engineering multiple rolling features at once
    # Percentiles and partial functions

# Chapter 4: Validating and Inspecting Time Series Models

    # Creating time-shifted features
    # Special case: Auto-regressive models
    # Visualize regression coefficients
    # Auto-regression with a smoother time series
    # Cross-validation with shuffling (don't do this for time series)
    # Cross-validation without shuffling 
    # Time-based cross-validation (do this for time series cross validation)
    # Bootstrapping a confidence interval
    # Calculating variability in model coefficients
    # Visualizing model score variability over time
    # Accounting for non-stationarity
    # Where to go next!!!!!!!!!!!!!
















######## Chapter 1: Time Series and Machine Learning Primer















### Plotting a time series (I)

# In this exercise, you'll practice plotting the values of two time series without 
# the time component. Two DataFrames, data and data2 are available in your workspace.
# Unless otherwise noted, assume that all required packages are loaded with their 
# common aliases throughout this course. Note: This course assumes some familiarity 
# with time series data, as well as how to use them in data analytics pipelines. For 
# an introduction to time series, we recommend the Introduction to Time Series Analysis 
# in Python and Visualizing Time Series Data with Python courses.

# Plot the time series in each dataset
fig, axs = plt.subplots(2, 1, figsize=(5, 10))
data.iloc[:1000].plot(y='data_values', ax=axs[0])
data2.iloc[:1000].plot(y='data_values', ax=axs[1])
plt.show()







### Plotting a time series (II)

# You'll now plot both the datasets again, but with the included time stamps for 
# each (stored in the column called "time"). Let's see if this gives you some more 
# context for understanding each time series data.

# Plot the time series in each dataset
fig, axs = plt.subplots(2, 1, figsize=(5, 10))
data.iloc[:1000].plot(x='time', y='data_values', ax=axs[0])
data2.iloc[:1000].plot(x='time', y='data_values', ax=axs[1])
plt.show()







### Fitting a simple model: classification

# In this exercise, you'll use the iris dataset (representing petal characteristics 
# of a number of flowers) to practice using the scikit-learn API to fit a 
# classification model. You can see a sample plot of the data to the right.

# Note: This course assumes some familiarity with Machine Learning and scikit-learn. 
# For an introduction to scikit-learn, we recommend the Supervised Learning with 
# Scikit-Learn and Preprocessing for Machine Learning in Python courses.

from sklearn.svm import LinearSVC

# Construct data for the model
X = data[["petal length (cm)", "petal width (cm)"]]
y = data[['target']]

# Fit the model
model = LinearSVC()
model.fit(X, y)







### Predicting using a classification model

# Now that you have fit your classifier, let's use it to predict the type of flower 
# (or class) for some newly-collected flowers. Information about petal width and 
# length for several new flowers is stored in the variable targets. Using the 
# classifier you fit, you'll predict the type of each flower.

# Create input array
X_predict = targets[['petal length (cm)', 'petal width (cm)']]

# Predict with the model
predictions = model.predict(X_predict)
print(predictions)

# Visualize predictions and actual values
plt.scatter(X_predict['petal length (cm)'], X_predict['petal width (cm)'],
            c=predictions, cmap=plt.cm.coolwarm)
plt.title("Predicted class values")
plt.show()

<script.py> output:
    [2 1 2 2 2 2 2 1 2 1 2 1 1 1 1 1 2 2 1 1]
    # Good job! Note that the output of your predictions are all integers, 
    # representing that datapoint's predicted class.







### Fitting a simple model: regression

# In this exercise, you'll practice fitting a regression model using data from 
# the Boston housing market. A DataFrame called boston is available in your 
# workspace. It contains many variables of data (stored as columns). Can you find a 
# relationship between the following two variables?

# "AGE": proportion of owner-occupied units built prior to 1940
# "RM" : average number of rooms per dwelling

from sklearn import linear_model

# Prepare input and output DataFrames
X = boston[['AGE']]
y = boston[['RM']]

# Fit the model
model = linear_model.LinearRegression()
model.fit(X, y)

Out[12]: 
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,
         normalize=False)







### Predicting using a regression model

# Now that you've fit a model with the Boston housing data, lets see what predictions 
# it generates on some new data. You can investigate the underlying relationship that 
# the model has found between inputs and outputs by feeding in a range of numbers as 
# inputs and seeing what the model predicts for each input.

# A 1-D array new_inputs consisting of 100 "new" values for "AGE" (proportion of 
# owner-occupied units built prior to 1940) is available in your workspace along 
# with the model you fit in the previous exercise.

# Generate predictions with the model using those inputs
predictions = model.predict(new_inputs.reshape(-1,1))

# Visualize the inputs and predicted values
plt.scatter(new_inputs, predictions, color='r', s=3)
plt.xlabel('inputs')
plt.ylabel('predictions')
plt.show()








### Inspecting the classification data

# In these final exercises of this chapter, you'll explore the two datasets you'll use 
# in this course. The first is a collection of heartbeat sounds. Hearts normally have a predictable 
# sound pattern as they beat, but some disorders can cause the heart to beat abnormally. 
# This dataset contains a training set with labels for each type of heartbeat, and a testing 
# set with no labels. You'll use the testing set to validate your models.
# As you have labeled data, this dataset is ideal for classification. In fact, it was 
# originally offered as a part of a public Kaggle competition.

import librosa as lr
from glob import glob

# List all the wav files in the folder
audio_files = glob(data_dir + '/*.wav')

# Read in the first audio file, create the time array
audio, sfreq = lr.load(audio_files[0])
time = np.arange(0, len(audio)) / sfreq

# Plot audio over time
fig, ax = plt.subplots()
ax.plot(time, audio)
ax.set(xlabel='Time (s)', ylabel='Sound Amplitude')
plt.show()

# see heart.svg







### Inspecting the regression data

# The next dataset contains information about company market value over several years 
# of time. This is one of the most popular kind of time series data used for regression. 
# If you can model the value of a company as it changes over time, you can make predictions 
# about where that company will be in the future. This dataset was also originally provided 
# as part of a public Kaggle competition.

# In this exercise, you'll plot the time series for a number of companies to get an 
# understanding of how they are (or aren't) related to one another.

# Read in the data
data = pd.read_csv('prices.csv', index_col=0)

# Convert the index of the DataFrame to datetime
data.index = pd.to_datetime(data.index)
print(data.head())

# Loop through each column, plot its values over time
fig, ax = plt.subplots()
for column in data:
    data[column].plot(ax=ax, label=column)
ax.legend()
plt.show()

# see stocks.svg















######## Chapter 2: Time Series as Inputs to a Model
















### Many repetitions of sounds

# In this exercise, you'll start with perhaps the simplest classification technique: 
# averaging across dimensions of a dataset and visually inspecting the result.
# You'll use the heartbeat data described in the last chapter. Some recordings are 
# normal heartbeat activity, while others are abnormal activity. Let's see if you 
# can spot the difference.
# Two DataFrames, normal and abnormal, each with the shape of (n_times_points, n_audio_files) 
# containing the audio for several heartbeats are available in your workspace. Also, 
# the sampling frequency is loaded into a variable called sfreq. A convenience plotting 
# function show_plot_and_make_titles() is also available in your workspace.

fig, axs = plt.subplots(3, 2, figsize=(15, 7), sharex=True, sharey=True)

# Calculate the time array
time = np.arange(len(normal)) / sfreq

# Stack the normal/abnormal audio so you can loop and plot
stacked_audio = np.hstack([normal, abnormal]).T

# Loop through each audio file / ax object and plot
# .T.ravel() transposes the array, then unravels it into a 1-D vector for looping
for iaudio, ax in zip(stacked_audio, axs.T.ravel()):
    ax.plot(time, iaudio)
show_plot_and_make_titles()

# see hearts.svg







### Invariance in time

# While you should always start by visualizing your raw data, this is often uninformative 
# when it comes to discriminating between two classes of data points. Data is usually 
# noisy or exhibits complex patterns that aren't discoverable by the naked eye.
# Another common technique to find simple differences between two sets of data is to 
# average across multiple instances of the same class. This may remove noise and 
# reveal underlying patterns (or, it may not).
# In this exercise, you'll average across many instances of each class of heartbeat sound.
# The two DataFrames (normal and abnormal) and the time array (time) from the 
# previous exercise are available in your workspace.

# Average across the audio files of each DataFrame
mean_normal = np.mean(normal, axis=1)
mean_abnormal = np.mean(abnormal, axis=1)

# Plot each average over time
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3), sharey=True)
ax1.plot(time, mean_normal)
ax1.set(title="Normal Data")
ax2.plot(time, mean_abnormal)
ax2.set(title="Abnormal Data")
plt.show()

# see mean_hearts.svg








### Build a classification model

# While eye-balling differences is a useful way to gain an intuition for the data, 
# let's see if you can operationalize things with a model. In this exercise, you will 
# use each repetition as a datapoint, and each moment in time as a feature to fit a 
# classifier that attempts to predict abnormal vs. normal heartbeats using only the 
# raw data.

# We've split the two DataFrames (normal and abnormal) into X_train, X_test, 
# y_train, and y_test.

from sklearn.svm import LinearSVC

# Initialize and fit the model
model = LinearSVC()
model.fit(X_train, y_train)

# Generate predictions and score them manually
predictions = model.predict(X_test)
print(sum(predictions == y_test.squeeze()) / len(y_test))

<script.py> output:
    0.555555555556

In [5]: predictions
Out[5]: 
array(['normal', 'normal', 'normal', 'normal', 'normal', 'normal',
       'normal', 'normal', 'abnormal', 'normal', 'normal', 'normal',
       'abnormal', 'normal', 'normal', 'abnormal', 'normal', 'normal'],
      dtype='<U8')

In [4]: y_test
Out[4]: 
array(['abnormal', 'abnormal', 'normal', 'abnormal', 'abnormal',
       'abnormal', 'abnormal', 'abnormal', 'abnormal', 'normal', 'normal',
       'abnormal', 'abnormal', 'normal', 'normal', 'abnormal', 'normal',
       'normal'],
      dtype='<U8')







### Calculating the envelope of sound

# One of the ways you can improve the features available to your model is to remove 
# some of the noise present in the data. In audio data, a common way to do this is 
# to smooth the data and then rectify it so that the total amount of sound energy 
# over time is more distinguishable. You'll do this in the current exercise.
# A heartbeat file is available in the variable audio.

# Plot the raw data first
audio.plot(figsize=(10, 5))
plt.show()

# see raw_audio.svg

# Rectify the audio signal
audio_rectified = audio.apply(np.abs)

# Plot the result
audio_rectified.plot(figsize=(10, 5))
plt.show()

# see rectified_audio.svg

# Smooth by applying a rolling mean
audio_rectified_smooth = audio_rectified.rolling(50).mean()

# Plot the result
audio_rectified_smooth.plot(figsize=(10, 5))
plt.show()

# see rolling_mean_audio.svg







### Calculating features from the envelope

# Now that you've removed some of the noisier fluctuations in the audio, let's 
# see if this improves your ability to classify.
# audio_rectified_smooth from the previous exercise is available in your workspace.

# Calculate stats
means = np.mean(audio_rectified_smooth, axis=0)
stds = np.std(audio_rectified_smooth, axis=0)
maxs = np.max(audio_rectified_smooth, axis=0)

# Create the X and y arrays
X = np.column_stack([means, stds, maxs])
y = labels.reshape([-1, 1])

# Fit the model and score on testing data
from sklearn.model_selection import cross_val_score
percent_score = cross_val_score(model, X, y, cv=5)
print(np.mean(percent_score))

<script.py> output:
    0.714219114219







### Derivative features: The tempogram

# One benefit of cleaning up your data is that it lets you compute more sophisticated 
# features. For example, the envelope calculation you performed is a common technique 
# in computing tempo and rhythm features. In this exercise, you'll use librosa to compute 
# some tempo and rhythm features for heartbeat data, and fit a model once more.

# Note that librosa functions tend to only operate on numpy arrays instead of DataFrames, so 
# we'll access our Pandas data as a Numpy array with the .values attribute.

# Calculate the tempo of the sounds
tempos = []
for col, i_audio in audio.items():
    tempos.append(lr.beat.tempo(i_audio.values, sr=sfreq, hop_length=2**6, aggregate=None))

# Convert the list to an array so you can manipulate it more easily
tempos = np.array(tempos)

# Calculate statistics of each tempo
tempos_mean = tempos.mean(axis=-1)
tempos_std = tempos.std(axis=-1)
tempos_max = tempos.max(axis=-1)

# Create the X and y arrays
X = np.column_stack([means, stds, maxs, tempos_mean, tempos_std, tempos_max])
y = labels.reshape([-1, 1])

# Fit the model and score on testing data
percent_score = cross_val_score(model, X, y, cv=5)
print(np.mean(percent_score))

<script.py> output:
    0.483216783217

# Note: score went down due to small sample size (supposedly)







### Spectrograms of heartbeat audio

# Spectral engineering is one of the most common techniques in machine learning for 
# time series data. The first step in this process is to calculate a spectrogram of 
# sound. This describes what spectral content (e.g., low and high pitches) are present 
# in the sound over time. In this exercise, you'll calculate a spectrogram of a heartbeat audio file.
# We've loaded a single heartbeat sound in the variable audio.

# Import the stft function
from librosa.core import stft

# Prepare the STFT
HOP_LENGTH = 2**4
spec = stft(audio, hop_length=HOP_LENGTH, n_fft=2**7)

from librosa.core import amplitude_to_db
from librosa.display import specshow

# Convert into decibels
spec_db = amplitude_to_db(spec)

# Compare the raw audio to the spectrogram of the audio
fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
axs[0].plot(time, audio)
specshow(spec_db, sr=sfreq, x_axis='time', y_axis='hz', hop_length=HOP_LENGTH)
plt.show()

# see spectrogram.svg (cool!)







### Engineering spectral features

# As you can probably tell, there is a lot more information in a spectrogram 
# compared to a raw audio file. By computing the spectral features, you have a much 
# better idea of what's going on. As such, there are all kinds of spectral features 
# that you can compute using the spectrogram as a base. In this exercise, you'll 
# look at a few of these features.
# The spectogram spec from the previous exercise is available in your workspace.

import librosa as lr

# Calculate the spectral centroid and bandwidth for the spectrogram
bandwidths = lr.feature.spectral_bandwidth(S=spec)[0]
centroids = lr.feature.spectral_centroid(S=spec)[0]

from librosa.core import amplitude_to_db
from librosa.display import specshow

# Convert spectrogram to decibels for visualization
spec_db = amplitude_to_db(spec)

# Display these features on top of the spectrogram
fig, ax = plt.subplots(figsize=(10, 5))
ax = specshow(spec_db, x_axis='time', y_axis='hz', hop_length=HOP_LENGTH)
ax.plot(times_spec, centroids)
ax.fill_between(times_spec, centroids - bandwidths / 2, centroids + bandwidths / 2, alpha=.5)
ax.set(ylim=[None, 6000])
plt.show()

# see spectral_centroid_bandwidth.svg (cool!)







### Combining many features in a classifier

# You've spent this lesson engineering many features from the audio data - some contain 
# information about how the audio changes in time, others contain information about the 
# spectral content that is present.

# The beauty of machine learning is that it can handle all of these features at the same 
# time. If there is different information present in each feature, it should improve the 
# classifier's ability to distinguish the types of audio. Note that this often requires 
# more advanced techniques such as regularization, which we'll cover in the next chapter.

# For the final exercise in the chapter, we've loaded many of the features that you 
# calculated before. Combine all of them into an array that can be fed into the 
# classifier, and see how it does.

# Loop through each spectrogram
bandwidths = []
centroids = []

for spec in spectrograms:
    # Calculate the mean spectral bandwidth
    this_mean_bandwidth = np.mean(lr.feature.spectral_bandwidth(S=spec))
    # Calculate the mean spectral centroid
    this_mean_centroid = np.mean(lr.feature.spectral_centroid(S=spec))
    # Collect the values
    bandwidths.append(this_mean_bandwidth)  
    centroids.append(this_mean_centroid)

# Create X and y arrays
X = np.column_stack([means, stds, maxs, tempo_mean, tempo_max, tempo_std, bandwidths, centroids])
y = labels.reshape([-1, 1])

# Fit the model and score on testing data
percent_score = cross_val_score(model, X, y, cv=5)
print(np.mean(percent_score))

<script.py> output:
    0.415034965035

# Good job! You calculated many different features of the audio, and combined each of 
# them under the assumption that they provide independent information that can be used 
# in classification. You may have noticed that the accuracy of your models varied a 
# lot when using different set of features. This chapter was focused on creating new 
# "features" from raw data and not obtaining the best accuracy. To improve the accuracy, 
# you want to find the right features that provide relevant information and also 
# build models on much larger data.















######## Chapter 3: Predicting Time Series Data
















### Introducing the dataset

# As mentioned in the video, you'll deal with stock market prices that fluctuate over 
# time. In this exercise you've got historical prices from two tech companies (Ebay and 
# Yahoo) in the DataFrame prices. You'll visualize the raw data for the two companies, 
# then generate a scatter plot showing how the values for each company compare with one 
# another. Finally, you'll add in a "time" dimension to your scatter plot so you can 
# see how this relationship changes over time.

# The data has been loaded into a DataFrame called prices.

# Plot the raw values over time
prices.plot()
plt.show()

# see prices.svg

# Scatterplot with one company per axis
prices.plot.scatter('EBAY', 'YHOO')
plt.show()

# see prices_scatter.svg

# Scatterplot with color relating to time
prices.plot.scatter('EBAY', 'YHOO', c=prices.index, 
                    cmap=plt.cm.viridis, colorbar=False)
plt.show()

# see prices_scatter_color.svg








### Fitting a simple regression model

# Now we'll look at a larger number of companies. Recall that we have historical 
# price values for many companies. Let's use data from several companies to predict 
# the value of a test company. You'll attempt to predict the value of the Apple 
# stock price using the values of NVidia, Ebay, and Yahoo. Each of these is stored 
# as a column in the all_prices DataFrame. Below is a mapping from company name to column name:
# ebay: "EBAY"
# nvidia: "NVDA"
# yahoo: "YHOO"
# apple: "AAPL"
# We'll use these columns to define the input/output arrays in our model.

from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Use stock symbols to extract training data
X = all_prices[['EBAY', 'NVDA', 'YHOO']]
y = all_prices[['AAPL']]

# Fit and score the model with cross-validation
scores = cross_val_score(Ridge(), X, y, cv=3)
print(scores)

<script.py> output:
    [-6.09050633 -0.3179172  -3.72957284]







# Visualizing predicted values

# When dealing with time series data, it's useful to visualize model predictions on 
# top of the "actual" values that are used to test the model.

# In this exercise, after splitting the data (stored in the variables X and y) into 
# training and test sets, you'll build a model and then visualize the model's 
# predictions on top of the testing data in order to estimate the model's performance.

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Split our data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    train_size=.8, shuffle=False, random_state=1)

# Fit our model and generate predictions
model = Ridge()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
score = r2_score(y_test, predictions)
print(score)

<script.py> output:
    -5.70939901949
# note: I don't understand this score as an r^2 value? see plot for further help,
# the ridge model is a poor estimator for the problem.

# Visualize our predictions along with the "true" values, and print the score
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(y_test, color='k', lw=3)
ax.plot(predictions, color='r', lw=2)
plt.show()

# see ridge_prediction.svg







### Visualizing messy data

# Let's take a look at a new dataset - this one is a bit less-clean than what you've 
# seen before. As always, you'll first start by visualizing the raw data. Take a close 
# look and try to find datapoints that could be problematic for fitting models.

# The data has been loaded into a DataFrame called prices.

# Visualize the dataset
prices.plot(legend=False)
plt.tight_layout()
plt.show()

# Count the missing values of each time series
missing_values = prices.isna().sum()
print(missing_values)

# see missing.svg







### Imputing missing values

# When you have missing data points, how can you fill them in?
# In this exercise, you'll practice using different interpolation methods to fill in 
# some missing values, visualizing the result each time. But first, you will create 
# the function (interpolate_and_plot()) you'll use to interpolate missing data 
# points and plot them. A single time series has been loaded into a DataFrame called prices.

# Create a function we'll use to interpolate and plot
def interpolate_and_plot(prices, interpolation):

    # Create a boolean mask for missing values
    missing_values = prices.isna()

    # Interpolate the missing values
    prices_interp = prices.interpolate(interpolation)

    # Plot the results, highlighting the interpolated values in black
    fig, ax = plt.subplots(figsize=(10, 5))
    prices_interp.plot(color='k', alpha=.6, ax=ax, legend=False)
    
    # Now plot the interpolated values on top in red
    prices_interp[missing_values].plot(ax=ax, color='r', lw=3, legend=False)
    plt.show()

# Interpolate using the latest non-missing value
interpolation_type = 'zero'
interpolate_and_plot(prices, interpolation_type)

# see interpolation1_zero.svg

# Interpolate using linear interpolation
interpolation_type = 'linear'
interpolate_and_plot(prices, interpolation_type)

# see interpolation2_linear.svg

# Interpolate with a quadratic function
interpolation_type = 'quadratic'
interpolate_and_plot(prices, interpolation_type)

# see interpolation3_quadratic.svg







### Transforming raw data

# In the last chapter, you calculated the rolling mean. In this exercise, you will 
# define a function that calculates the percent change of the latest data point from 
# the mean of a window of previous data points. This function will help you calculate 
# the percent change over a rolling window.

# This is a more stable kind of time series that is often useful in machine learning.

# Your custom function
def percent_change(series):
    # Collect all *but* the last value of this window, then the final value
    previous_values = series[:-1]
    last_value = series[-1]

    # Calculate the % difference between the last value and the mean of earlier values
    percent_change = (last_value - np.mean(previous_values)) / np.mean(previous_values)
    return percent_change

# Apply your custom function and plot
prices_perc = prices.rolling(20).apply(percent_change)
prices_perc.loc["2014":"2015"].plot()
plt.show()

# see transforming_data.svg







### Handling outliers

# In this exercise, you'll handle outliers - data points that are so different from 
# the rest of your data, that you treat them differently from other "normal-looking" 
# data points. You'll use the output from the previous exercise (percent change over 
# time) to detect the outliers. First you will write a function that replaces outlier 
# data points with the median value from the entire time series.

def replace_outliers(series):
    # Calculate the absolute difference of each timepoint from the series mean
    absolute_differences_from_mean = np.abs(series - np.mean(series))
    
    # Calculate a mask for the differences that are > 3 standard deviations from zero
    this_mask = absolute_differences_from_mean > (np.std(series) * 3)
    
    # Replace these values with the median accross the data
    series[this_mask] = np.nanmedian(series)
    return series

# Apply your preprocessing function to the timeseries and plot the results
prices_perc = prices_perc.apply(replace_outliers)
prices_perc.loc["2014":"2015"].plot()
plt.show()

# see replace_outliers.svg







### Engineering multiple rolling features at once

# Now that you've practiced some simple feature engineering, let's move on to 
# something more complex. You'll calculate a collection of features for your time 
# series data and visualize what they look like over time. This process resembles 
# how many other time series models operate.

# Define a rolling window with Pandas, excluding the right-most datapoint of the window
prices_perc_rolling = prices_perc.rolling(20, min_periods=5, closed='right')

# Define the features you'll calculate for each window
features_to_calculate = [np.min, np.max, np.mean, np.std]

# Calculate these features for your rolling window object
features = prices_perc_rolling.aggregate(features_to_calculate)

# Plot the results
ax = features.loc[:"2011-01"].plot()
prices_perc.loc[:"2011-01"].plot(ax=ax, color='k', alpha=.2, lw=3)
ax.legend(loc=(1.01, .6))
plt.show()

# see aggregate_func.svg







### Percentiles and partial functions

# In this exercise, you'll practice how to pre-choose arguments of a function so that 
# you can pre-configure how it runs. You'll use this to calculate several percentiles 
# of your data using the same percentile() function in numpy.

# Import partial from functools
from functools import partial
percentiles = [1, 10, 25, 50, 75, 90, 99]

# Use a list comprehension to create a partial function for each quantile
percentile_functions = [partial(np.percentile, q=percentile) for percentile in percentiles]

# Calculate each of these quantiles on the data using a rolling window
prices_perc_rolling = prices_perc.rolling(20, min_periods=5, closed='right')
features_percentiles = prices_perc_rolling.aggregate(percentile_functions)

# Plot a subset of the result
ax = features_percentiles.loc[:"2011-01"].plot(cmap=plt.cm.viridis)
ax.legend(percentiles, loc=(1.01, .5))
plt.show()

# see partial_func.svg







### Using "date" information

# It's easy to think of timestamps as pure numbers, but don't forget they generally 
# correspond to things that happen in the real world. That means there's often extra 
# information encoded in the data such as "is it a weekday?" or "is it a holiday?". 
# This information is often useful in predicting timeseries data.

# In this exercise, you'll extract these date/time based features. A single time 
# series has been loaded in a variable called prices.

# Extract date features from the data, add them as columns
prices_perc['day_of_week'] = prices_perc.index.dayofweek
prices_perc['week_of_year'] = prices_perc.index.weekofyear
prices_perc['month_of_year'] = prices_perc.index.month

# Print prices_perc
print(prices_perc)

# output:
EBAY  day_of_week  week_of_year  month_of_year
date                                                          
2014-01-02  0.017938            3             1              1
2014-01-03  0.002268            4             1              1
2014-01-06 -0.027365            0             2              1
2014-01-07 -0.006665            1             2              1
2014-01-08 -0.017206            2             2              1















####### Chapter 4: Validating and Inspecting Time Series Models















### Creating time-shifted features

# In machine learning for time series, it's common to use information about previous 
# time points to predict a subsequent time point.

# In this exercise, you'll "shift" your raw data and visualize the results. You'll 
# use the percent change time series that you calculated in the previous chapter, 
# this time with a very short window. A short window is important because, in a 
# real-world scenario, you want to predict the day-to-day fluctuations of a time 
# series, not its change over a longer window of time.

# These are the "time lags"
shifts = np.arange(1, 11).astype(int)

# Use a dictionary comprehension to create name: value pairs, one pair per shift
shifted_data = {"lag_{}_day".format(day_shift): prices_perc.shift(day_shift) for day_shift in shifts}

# Convert into a DataFrame for subsequent use
prices_perc_shifted = pd.DataFrame(shifted_data)

# Plot the first 100 samples of each
ax = prices_perc_shifted.iloc[:100].plot(cmap=plt.cm.viridis)
prices_perc.iloc[:100].plot(color='r', lw=2)
ax.legend(loc='best')
plt.show()

# see shifted.svg







### Special case: Auto-regressive models

# Now that you've created time-shifted versions of a single time series, you can fit 
# an auto-regressive model. This is a regression model where the input features are 
# time-shifted versions of the output time series data. You are using previous values 
# of a timeseries to predict current values of the same timeseries (thus, it is auto-regressive).

# By investigating the coefficients of this model, you can explore any repetitive 
# patterns that exist in a timeseries, and get an idea for how far in the past a 
# data point is predictive of the future.

# Replace missing values with the median for each column
X = prices_perc_shifted.fillna(np.nanmedian(prices_perc_shifted))
y = prices_perc.fillna(np.nanmedian(prices_perc))

# Fit the model
model = Ridge()
model.fit(X, y)







### Visualize regression coefficients

# Now that you've fit the model, let's visualize its coefficients. This is an important 
# part of machine learning because it gives you an idea for how the different features 
# of a model affect the outcome.

# The shifted time series DataFrame (prices_perc_shifted) and the regression model 
# (model) are available in your workspace.

# In this exercise, you will create a function that, given a set of coefficients and 
# feature names, visualizes the coefficient values.

def visualize_coefficients(coefs, names, ax):
    # Make a bar plot for the coefficients, including their names on the x-axis
    ax.bar(names, coefs)
    ax.set(xlabel='Coefficient name', ylabel='Coefficient value')
    
    # Set formatting so it looks nice
    plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    return ax

# Visualize the output data up to "2011-01"
fig, axs = plt.subplots(2, 1, figsize=(10, 5))
y.loc[:'2011-01'].plot(ax=axs[0])

# Run the function to visualize model's coefficients
visualize_coefficients(model.coef_, prices_perc_shifted.columns, ax=axs[1])
plt.show()

# see lagged_coef_correlations.svg

# Note: When you use time-lagged features on the raw data, you see that the highest 
# coefficient by far is the first one. This means that the N-1th time point is useful 
# in predicting the Nth timepoint, but no other points are useful.







### Auto-regression with a smoother time series

# Now, let's re-run the same procedure using a smoother signal. You'll use the same 
# percent change algorithm as before, but this time use a much larger window (40 
# instead of 20). As the window grows, the difference between neighboring timepoints 
# gets smaller, resulting in a smoother signal. What do you think this will do to 
# the auto-regressive model?

# prices_perc_shifted and model (updated to use a window of 40) are available in your workspace.

In [3]: prices_perc_shifted
            lag_1_day  lag_2_day  lag_3_day  lag_4_day  lag_5_day  lag_6_day  \
date                                                                           
2010-01-04        NaN        NaN        NaN        NaN        NaN        NaN   
2010-01-05        NaN        NaN        NaN        NaN        NaN        NaN   
2010-01-06        NaN        NaN        NaN        NaN        NaN        NaN   
2010-01-07        NaN        NaN        NaN        NaN        NaN        NaN   
2010-01-08        NaN        NaN        NaN        NaN        NaN        NaN   
2010-01-11        NaN        NaN        NaN        NaN        NaN        NaN   
2010-01-12        NaN        NaN        NaN        NaN        NaN        NaN   
2010-01-13        NaN        NaN        NaN        NaN        NaN        NaN   
2010-01-14        NaN        NaN        NaN        NaN        NaN        NaN   
2010-01-15        NaN        NaN        NaN        NaN        NaN        NaN   
2010-01-19        NaN        NaN        NaN        NaN        NaN        NaN   
2010-01-20        NaN        NaN        NaN        NaN        NaN        NaN   
2010-01-21        NaN        NaN        NaN        NaN        NaN        NaN   
2010-01-22        NaN        NaN        NaN        NaN        NaN        NaN   
2010-01-25        NaN        NaN        NaN        NaN        NaN        NaN   
2010-01-26        NaN        NaN        NaN        NaN        NaN        NaN   
2010-01-27        NaN        NaN        NaN        NaN        NaN        NaN   
2010-01-28        NaN        NaN        NaN        NaN        NaN        NaN   
2010-01-29        NaN        NaN        NaN        NaN        NaN        NaN   
2010-02-01        NaN        NaN        NaN        NaN        NaN        NaN   
2010-02-02        NaN        NaN        NaN        NaN        NaN        NaN   
2010-02-03        NaN        NaN        NaN        NaN        NaN        NaN   
2010-02-04        NaN        NaN        NaN        NaN        NaN        NaN   
2010-02-05        NaN        NaN        NaN        NaN        NaN        NaN   
2010-02-08        NaN        NaN        NaN        NaN        NaN        NaN   
2010-02-09        NaN        NaN        NaN        NaN        NaN        NaN   
2010-02-10        NaN        NaN        NaN        NaN        NaN        NaN   
2010-02-11        NaN        NaN        NaN        NaN        NaN        NaN   
2010-02-12        NaN        NaN        NaN        NaN        NaN        NaN   
2010-02-16        NaN        NaN        NaN        NaN        NaN        NaN   
...               ...        ...        ...        ...        ...        ...   
2016-11-17  -0.030700  -0.057452  -0.071418  -0.048629  -0.055759  -0.029712   
2016-11-18  -0.030037  -0.030700  -0.057452  -0.071418  -0.048629  -0.055759   
2016-11-21  -0.028460  -0.030037  -0.030700  -0.057452  -0.071418  -0.048629   
2016-11-22  -0.013089  -0.028460  -0.030037  -0.030700  -0.057452  -0.071418   
2016-11-23  -0.012166  -0.013089  -0.028460  -0.030037  -0.030700  -0.057452   
2016-11-25  -0.016723  -0.012166  -0.013089  -0.028460  -0.030037  -0.030700   
2016-11-28  -0.011560  -0.016723  -0.012166  -0.013089  -0.028460  -0.030037   
2016-11-29  -0.013224  -0.011560  -0.016723  -0.012166  -0.013089  -0.028460   
2016-11-30  -0.013984  -0.013224  -0.011560  -0.016723  -0.012166  -0.013089   
2016-12-01  -0.021958  -0.013984  -0.013224  -0.011560  -0.016723  -0.012166   
2016-12-02  -0.030516  -0.021958  -0.013984  -0.013224  -0.011560  -0.016723   
2016-12-05  -0.025913  -0.030516  -0.021958  -0.013984  -0.013224  -0.011560   
2016-12-06  -0.032000  -0.025913  -0.030516  -0.021958  -0.013984  -0.013224 

# Visualize the output data up to "2011-01"
fig, axs = plt.subplots(2, 1, figsize=(10, 5))
y.loc[:'2011-01'].plot(ax=axs[0])

# Run the function to visualize model's coefficients
visualize_coefficients(model.coef_, prices_perc_shifted.columns, ax=axs[1])
plt.show()

# Correct! As you can see here, by transforming your data with a larger window, you've 
# also changed the relationship between each timepoint and the ones that come just before 
# it. This model's coefficients gradually go down to zero, which means that the signal 
# itself is smoother over time. Be careful when you see something like this, as it 
# means your data is not i.i.d. (independent, identically distributed.  ie time series data != i.i.d)

# see smoothed_autocorrelation.svg







### Cross-validation with shuffling (don't do this for time series)

# As you'll recall, cross-validation is the process of splitting your data into training 
# and test sets multiple times. Each time you do this, you choose a different training 
# and test set. In this exercise, you'll perform a traditional ShuffleSplit cross-validation 
# on the company value data from earlier. Later we'll cover what changes need to be made 
# for time series data. The data we'll use is the same historical price data for several 
# large companies.

# An instance of the Linear regression object (model) is available in your workspace 
# along with the function r2_score() for scoring. Also, the data is stored in arrays 
# X and y. We've also provided a helper function (visualize_predictions()) to help 
# visualize the results.

# Import ShuffleSplit and create the cross-validation object
from sklearn.model_selection import ShuffleSplit
cv = ShuffleSplit(n_splits=10, random_state=1)

# Iterate through CV splits
results = []
for tr, tt in cv.split(X, y):
    # Fit the model on training data
    model.fit(X[tr], y[tr])
    
    # Generate predictions on the test data, score the predictions, and collect
    prediction = model.predict(X[tt])
    score = r2_score(y[tt], prediction)
    results.append((prediction, score, tt))

# Custom function to quickly visualize predictions
visualize_predictions(results)







### Cross-validation without shuffling 

# Now, re-run your model fit using block cross-validation (without shuffling all 
# datapoints). In this case, neighboring time-points will be kept close to one another. 
# How do you think the model predictions will look in each cross-validation loop?

# An instance of the Linear regression model object is available in your workspace. 
# Also, the arrays X and y (training data) are available too.

# Create KFold cross-validation object
from sklearn.model_selection import KFold
cv = KFold(n_splits=10, shuffle=False, random_state=1)

# Iterate through CV splits
results = []
for tr, tt in cv.split(X, y):
    # Fit the model on training data
    model.fit(X[tr],y[tr])
    
    # Generate predictions on the test data and collect
    prediction = model.predict(X[tt])
    results.append((prediction, tt))
    
# Custom function to quickly visualize predictions
visualize_predictions(results)

see KFold_split.svg








### Time-based cross-validation (do this for time series cross validation)

# Finally, let's visualize the behavior of the time series cross-validation iterator 
# in scikit-learn. Use this object to iterate through your data one last time, 
# visualizing the training data used to fit the model on each iteration.

# An instance of the Linear regression model object is available in your workpsace. 
# Also, the arrays X and y (training data) are available too.

# Import TimeSeriesSplit
from sklearn.model_selection import TimeSeriesSplit

# Create time-series cross-validation object
cv = TimeSeriesSplit(n_splits=10)

# Iterate through CV splits
fig, ax = plt.subplots()
for ii, (tr, tt) in enumerate(cv.split(X, y)):
    # Plot the training data on each iteration, to see the behavior of the CV
    ax.plot(tr, ii + y[tr])

ax.set(title='Training data on each CV iteration', ylabel='CV iteration')
plt.show()

see timeSeriesSplit.svg







### Bootstrapping a confidence interval

# A useful tool for assessing the variability of some data is the bootstrap. In this 
# exercise, you'll write your own bootstrapping function that can be used to return a 
# bootstrapped confidence interval.

# This function takes three parameters: a 2-D array of numbers (data), a list of 
# percentiles to calculate (percentiles), and the number of boostrap iterations to 
# use (n_boots). It uses the resample function to generate a bootstrap sample, and 
# then repeats this many times to calculate the confidence interval.

from sklearn.utils import resample

def bootstrap_interval(data, percentiles=(2.5, 97.5), n_boots=100):
    """Bootstrap a confidence interval for the mean of columns of a 2-D dataset."""
    # Create our empty array to fill the results
    bootstrap_means = np.zeros([n_boots, data.shape[-1]])
    for ii in range(n_boots):
        # Generate random indices for our data *with* replacement, then take the sample mean
        random_sample = resample(data)
        bootstrap_means[ii] = random_sample.mean(axis=0)
        
    # Compute the percentiles of choice for the bootstrapped means
    percentiles = np.percentile(bootstrap_means, percentiles, axis=0)
    return percentiles







### Calculating variability in model coefficients

# In this lesson, you'll re-run the cross-validation routine used before, but this 
# time paying attention to the model's stability over time. You'll investigate the 
# coefficients of the model, as well as the uncertainty in its predictions.

# Begin by assessing the stability (or uncertainty) of a model's coefficients across 
# multiple CV splits. Remember, the coefficients are a reflection of the pattern 
# that your model has found in the data.

# An instance of the Linear regression object (model) is available in your workpsace. 
# Also, the arrays X and y (the data) are available too.

# Iterate through CV splits
n_splits = 100
cv = TimeSeriesSplit(n_splits=n_splits)

# Create empty array to collect coefficients
coefficients = np.zeros([n_splits, X.shape[1]])

for ii, (tr, tt) in enumerate(cv.split(X, y)):
    # Fit the model on training data and collect the coefficients
    model.fit(X[tr], y[tr])
    coefficients[ii] = model.coef_

# Iterate through CV splits
n_splits = 100
cv = TimeSeriesSplit(n_splits=n_splits)

# Create empty array to collect coefficients
coefficients = np.zeros([n_splits, X.shape[1]])

for ii, (tr, tt) in enumerate(cv.split(X, y)):
    # Fit the model on training data and collect the coefficients
    model.fit(X[tr], y[tr])
    coefficients[ii] = model.coef_

see bootstrap_coefficients_stability.svg







### Visualizing model score variability over time

# Now that you've assessed the variability of each coefficient, let's do the same for 
# the performance (scores) of the model. Recall that the TimeSeriesSplit object will 
# use successively-later indices for each test set. This means that you can treat the 
# scores of your validation as a time series. You can visualize this over time in 
# order to see how the model's performance changes over time.

# An instance of the Linear regression model object is stored in model, a 
# cross-validation object in cv, and data in X and y.

from sklearn.model_selection import cross_val_score

# Generate scores for each split to see how the model performs over time
scores = cross_val_score(model, X, y, cv=cv, scoring=my_pearsonr)

# Convert to a Pandas Series object
scores_series = pd.Series(scores, index=times_scores, name='score')

# Bootstrap a rolling confidence interval for the mean score
scores_lo = scores_series.rolling(20).aggregate(partial(bootstrap_interval, percentiles=2.5))
scores_hi = scores_series.rolling(20).aggregate(partial(bootstrap_interval, percentiles=97.5))

# Plot the results
fig, ax = plt.subplots()
scores_lo.plot(ax=ax, label="Lower confidence interval")
scores_hi.plot(ax=ax, label="Upper confidence interval")
ax.legend()
plt.show()

see rolling_scores.svg







### Accounting for non-stationarity

# In this exercise, you will again visualize the variations in model scores, but now 
# for data that changes its statistics over time.

# An instance of the Linear regression model object is stored in model, a cross-validation 
# object in cv, and the data in X and y.

# Pre-initialize window sizes
window_sizes = [25, 50, 75, 100]

# Create an empty DataFrame to collect the stores
all_scores = pd.DataFrame(index=times_scores)

# Generate scores for each split to see how the model performs over time
for window in window_sizes:
    # Create cross-validation object using a limited lookback window
    cv = TimeSeriesSplit(n_splits=100, max_train_size=window)
    
    # Calculate scores across all CV splits and collect them in a DataFrame
    this_scores = cross_val_score(model, X, y, cv=cv, scoring=my_pearsonr)
    all_scores['Length {}'.format(window)] = this_scores

# Visualize the scores
ax = all_scores.rolling(10).mean().plot(cmap=plt.cm.coolwarm)
ax.set(title='Scores for multiple windows', ylabel='Correlation (r)')
plt.show()

see non_stationarity_results.svg







### # Where to go next!!!!!!!!!!!!!

-Advanced Window functions
-Signal Processing and filtering details
-Spectral Analysis

-tsfresh python package for calculating dozens of features from time series
-production pipelines for machine learning
-Quantopian for financial forecasting and model testing







