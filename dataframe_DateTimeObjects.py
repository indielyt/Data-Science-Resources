### FROM DATACAMP, PANDAS FOUNDATIONS COURSE

# Import Plotting Modules
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



### CREATE DATE TIME OBJECTS FROM LISTS


 # Prepare a format string: time_format (will be passed into pd.to_datetime)
time_format = '%Y-%m-%d %H:%M'

# Convert date_list into a datetime object: my_datetimes 
my_datetimes = pd.to_datetime(date_list, format=time_format)  

# Construct a pandas Series using temperature_list and my_datetimes: time_series
time_series = pd.Series(temperature_list, index=my_datetimes)




###  Partial String Indexing and Slicing


# In this exercise, a time series that contains hourly weather data has been pre-loaded for you. 
# This data was read using the parse_dates=True option in read_csv() with index_col="Dates" 
# so that the Index is indeed a DatetimeIndex.
# # All data from the 'Temperature' column has been extracted into the variable ts0. Your job 
# is to use a variety of natural date strings to extract one or more values from ts0.

# Extract the hour from 9pm to 10pm on '2010-10-11': ts1
ts1 = ts0.loc['2010-10-11 21:00:00']

# Extract '2010-07-04' from ts0: ts2
ts2 = ts0.loc['2010-7-4']
print (ts2)
# Extract data from '2010-12-15' to '2010-12-31': ts3
ts3 = ts0.loc['2010-12-15':'2010-12-31']
print (ts3)

# Extract the first temperature from ts3
ts4 = ts3.iloc[0]
print (ts4)

# Extract the last temperature from ts3
ts5 = ts3.iloc[-1]
print (ts5)




### REINDEXING BY MATCHING ONE DATAFRAMES INDEX TO ANOTHER

# Reindexing is useful in preparation for adding or otherwise combining two time series data sets. 
# To reindex the data, we provide a new index and ask pandas to try and match the old data to the 
# new index. If data is unavailable for one of the new index dates or times, you must tell pandas 
# how to fill it in. Otherwise, pandas will fill with NaN by default.
# # In this exercise, two time series data sets containing daily data have been pre-loaded for you, 
# each indexed by dates. The first, ts1, includes weekends, but the second, ts2, does not. The goal 
# is to combine the two data sets in a sensible way. Your job is to reindex the second data set so 
# that it has weekends as well, and then add it to the first. When you are done, it would be 
# informative to inspect your results.

# Reindex without fill method:  Where index of ts2 doesn't have data for ts1 index locations 
# (weekends in this case), 'NAN' is filled
ts3 = ts2.reindex(ts1.index)

# Reindex with fill method, using forward fill: Where index of ts2 doesn't have data for ts1 index 
# locations (weekends in this case), Friday's data is forward filled into the missing index locations
ts4 = ts2.reindex(ts1.index,method='ffill')





###  RESAMPLING WITH A DIFFERENT FREQUENCY

# Pandas provides methods for resampling time series data. When downsampling or upsampling, the syntax 
# is similar, but the methods called are different. Both use the concept of 
# 'method chaining' - df.method1().method2().method3() - to direct the output from one method 
# call to the input of the next, and so on, as a sequence of operations, one feeding into the next.
# For example, if you have hourly data, and just need daily data, pandas will not guess how to throw 
# out the 23 of 24 points. You must specify this in the method. One approach, for instance, could be 
# to take the mean, as in df.resample('D').mean().
# In this exercise, a data set containing hourly temperature data has been pre-loaded for you. 
# Your job is to resample the data using a variety of aggregation methods to answer a few questions.

# Downsample to 6 hour data and aggregate by mean: df1
df1 = df['Temperature'].resample('6h').mean()

# Downsample to daily data and count the number of data points: df2
df2 = df['Temperature'].resample('D').count()




### METHOD CHAINING AND DATE TIME OBJECTS / PANDAS DATAFRAMES

# Extract temperature data for August: august
august = df['Temperature'].loc['2010-08-01':'2010-08-31']

# Downsample to obtain only the daily highest temperatures in August: august_highs
august_highs = august.resample('D').max()

# Extract temperature data for February: february
february = df['Temperature'].loc['2010-02-01':'2010-02-28']

# Downsample to obtain the daily lowest temperatures in February: february_lows
february_lows = february.resample('D').min()

# Resample the August 2011 temperatures in df_clean by day and aggregate the maximum value: august_2011
august_2011 = df_clean.loc['2011-Aug','Temperatures'].resample('d').max()




### ROLLING MEAN AND FREQUENCY

# In this exercise, some hourly weather data is pre-loaded for you. You will continue to practice 
# resampling, this time using rolling means.  Rolling means (or moving averages) are generally 
# used to smooth out short-term fluctuations in time series data and highlight long-term trends. Y
# ou can read more about them here.  To use the .rolling() method, you must always use method 
# chaining, first calling .rolling() and then chaining an aggregation method after it. For example, 
# with a Series hourly_data, hourly_data.rolling(window=24).mean() would compute new values for each 
# hourly point, based on a 24-hour window stretching out behind each point. The frequency of the 
# output data is the same: it is still hourly. Such an operation is useful for smoothing time series data.
# Your job is to resample the data using the combination of .rolling() and .mean(). You will work 
# with the same DataFrame df from the previous exercise.

# Extract data from 2010-Aug-01 to 2010-Aug-15: unsmoothed
unsmoothed = df['Temperature']['2010-Aug-01':'2010-Aug-15']

# Apply a rolling mean with a 24 hour window: smoothed
smoothed = unsmoothed.rolling(window=24).mean()

# Create a new DataFrame with columns smoothed and unsmoothed: august
august = pd.DataFrame({'smoothed':smoothed, 'unsmoothed':unsmoothed})

# Plot both smoothed and unsmoothed data using august.plot().
august.plot(subplots=True)
plt.show()




###  METHOD CHAINING AND CLEANING DATA / EXTRACTING DATA

# Use .str.strip() to strip extra whitespace from df.columns. 
# Assign the result back to df.columns. In the 'Destination Airport' column, 
# extract all entries where Dallas ('DAL') is the destination airport. 
# Use .str.contains('DAL') for this and store the result in dallas.
# Resample dallas such that you get the total number of departures each day. 
# Store the result in daily_departures.
# Generate summary statistics for daily Dallas departures using .describe(). 
# Store the result in stats.

 # Strip extra whitespace from the column names: df.columns
df.columns = df.columns.str.strip()

# Extract data for which the destination airport is Dallas: dallas
dallas = df['Destination Airport'].str.contains('DAL')

# Compute the total number of Dallas departures each day: daily_departures
daily_departures = dallas.resample('D').sum()

# Generate the summary statistics for daily Dallas departures: stats
stats = daily_departures.describe()

# Extract multiple columns simultaneously
origin_destination = df[['Origin Airport', 'Destination Airportdf']]





###  TIME ZONE CONVERSION WITH DATE TIME OBJECTS

# Time zone handling with pandas typically assumes that you are handling the Index 
# of the Series. In this exercise, you will learn how to handle timezones that are 
# associated with datetimes in the column data, and not just the Index.
# You will work with the flight departure dataset again, and this time you will 
# select Los Angeles ('LAX') as the destination airport.  Here we will use a mask 
# to ensure that we only compute on data we actually want. 
# https://docs.scipy.org/doc/numpy/reference/maskedarray.generic.html

# Build a Boolean mask to filter out all the 'LAX' departure flights: mask
mask = df['Destination Airport'] == 'LAX'

# Use the mask to subset the data: la
la = df[mask]

# Combine two columns of data to create a datetime series: times_tz_none 
times_tz_none = pd.to_datetime(la['Date (MM/DD/YYYY)'] + ' ' + la['Wheels-off Time'])

# Localize the time to US/Central: times_tz_central
times_tz_central = times_tz_none.dt.tz_localize('US/Central')

# Convert the datetimes from US/Central to US/Pacific
times_tz_pacific = times_tz_central.dt.tz_convert('US/Pacific')




###  PLOTTING TIME SERIES

# Use pd.to_datetime() to convert the 'Date' column to a collection of 
# datetime objects, and assign back to df.Date.Set the index to this 
# updated 'Date' column, using df.set_index() with the optional keyword 
# argument inplace=True, so that you don't have to assign the result back to df.

# Convert the 'Date' column into a collection of datetime objects: df.Date
df.Date = pd.to_datetime(df.Date)

# Set the index to be the converted 'Date' column
df.set_index('Date', inplace=True)

# Re-plot the DataFrame to see that the axis is now datetime aware!
df.plot()
plt.show()

# Plot the summer data
df.Temperature['2010-Jun':'2010-Aug'].plot()
plt.show()
plt.clf()

# Plot the one week data
df.Temperature['2010-06-10':'2010-06-17'].plot()
plt.show()
plt.clf()


### CLEANING UP A MESSY DATAFRAME INTO DATE TIME INDEXED DATAFRAME

# Convert the date column to string: df_dropped['date']
df_dropped['date'] = df_dropped['date'].astype(str)

# Pad leading zeros to the Time column: df_dropped['Time']
df_dropped['Time'] = df_dropped['Time'].apply(lambda x:'{:0>4}'.format(x))

# Concatenate the new date and Time columns: date_string
date_string = df_dropped['date'] + df_dropped['Time']

# Convert the date_string Series to datetime: date_times
date_times = pd.to_datetime(date_string, format='%Y%m%d%H%M')

# Set the index to be the new date_times container: df_clean
df_clean = df_dropped.set_index(date_times)




### CLEANING UP NUMERIC COLUMNS

# The numeric columns contain missing values labeled as 'M'. In this exercise, 
# your job is to transform these columns such that they contain only numeric 
# values and interpret missing data as NaN.  The pandas function 
# pd.to_numeric() is ideal for this purpose: It converts a Series of values 
# to floating-point values. Furthermore, by specifying the keyword argument 
# errors='coerce', you can force strings like 'M' to be interpreted as NaN.

# Print the dry_bulb_faren temperature between 8 AM and 9 AM on June 20, 2011
print(df_clean.loc['2011-6-20 8:00:00':'2011-6-20 9:00:00', 'dry_bulb_faren'])

# Convert the dry_bulb_faren column to numeric values: df_clean['dry_bulb_faren']
df_clean['dry_bulb_faren'] = pd.to_numeric(df_clean['dry_bulb_faren'], errors='coerce')

# Print the transformed dry_bulb_faren temperature between 8 AM and 9 AM on June 20, 2011
print(df_clean.loc['2011-6-20 8:00:00': '2011-6-20 9:00:00', 'dry_bulb_faren'])

# Convert the wind_speed and dew_point_faren columns to numeric values
df_clean['wind_speed'] = pd.to_numeric(df_clean['wind_speed'], errors='coerce')
df_clean['dew_point_faren'] = pd.to_numeric(df_clean['dew_point_faren'], errors='coerce')



### COMPARING DATAFRAMES WITH DIFFERENT INDEXES

# Notice that the indexes of df_clean and df_climate are not aligned - df_clean has 
# dates in 2011, while df_climate has dates in 2010. This is why you extract the 
# temperature columns as NumPy arrays. An alternative approach is to use the 
# pandas .reset_index() method to make sure the Series align properly. You will 
# practice this approach as well.

# Downsample df_clean by day and aggregate by mean: daily_mean_2011
daily_mean_2011 = df_clean.resample('D').mean()

# Extract the dry_bulb_faren column from daily_mean_2011 using .values: daily_temp_2011
daily_temp_2011 = daily_mean_2011['dry_bulb_faren'].values

# Downsample df_climate by day and aggregate by mean: daily_climate
daily_climate = df_climate.resample('D').mean()

# Extract the Temperature column from daily_climate using .reset_index(): daily_temp_climate
daily_temp_climate = daily_climate.reset_index()['Temperature']

# Compute the difference between the two arrays and print the mean difference
difference = daily_temp_2011 - daily_temp_climate
print(difference.mean())

