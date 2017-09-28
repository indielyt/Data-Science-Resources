### DATACAMP - Pandas Foundations

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

### Read/Write from CSV

		# Read in csv
		# Clean up messy csv / export to csv or excel

### More basics

### Exploratory Data Analysis

		# MIN, MAX, MEAN
		# QUANTILES 
		# FILTERING DATA BY COLUMN
		# INDEXING AND SELECTING:   .iloc / .loc / .ix

### Plotting with Dataframes

		# BASIC LINE PLOT
		# MULTI-LINE PLOT
		# SCATTER PLOT
		# BOX & WHISKER PLOTS
		# SLICING DATAFRAME AND MAKING SUBPLOTS
		# PDF AND CFD PLOTS

### Date Time Objects

		# CREATE DATE TIME OBJECTS FROM LISTS
		# Partial String Indexing and Slicing
		# REINDEXING BY MATCHING ONE DATAFRAMES INDEX TO ANOTHER
		# RESAMPLING WITH A DIFFERENT FREQUENCY
		# METHOD CHAINING AND DATE TIME OBJECTS / PANDAS DATAFRAMES
		# ROLLING MEAN AND FREQUENCY
		# METHOD CHAINING AND CLEANING DATA / EXTRACTING DATA
		# TIME ZONE CONVERSION WITH DATE TIME OBJECTS
		# PLOTTING TIME SERIES
		# CLEANING UP A MESSY DATAFRAME INTO DATE TIME INDEXED DATAFRAME
		# CLEANING UP NUMERIC COLUMNS
		# COMPARING DATAFRAMES WITH DIFFERENT INDEXES


###################



# Read in the file: df1
df1 = pd.read_csv('world_population.csv')

# Create a list of the new column labels: new_labels
new_labels = ['year','population']

# Read in the file, specifying the header and names parameters: df2
df2 = pd.read_csv('world_population.csv', header=0, names=new_labels)

# Print both the DataFrames
print(df1)
print(df2)



####  Clean up a messy csv file with the .read_csv() methods, then save to csv and excel
# Read the raw file as-is: df1
df1 = pd.read_csv(file_messy)

# Print the output of df1.head()
print(df1.head())

# Read in the file with the correct parameters: df2
df2 = pd.read_csv(file_messy, delimiter=' ', header=3, comment='#')

# Print the output of df2.head()
print(df2.head())

# Save the cleaned up DataFrame to a CSV file without the index
df2.to_csv(file_clean, index=False)

# Save the cleaned up DataFrame to an excel file without the index
df2.to_excel('file_clean.xlsx', index=False)




###############





df = pd.DataFrame({
    'UNIT': ['R051', 'R079', 'R051', 'R079', 'R051', 'R079', 'R051', 'R079', 'R051'],
    'TIMEn': ['00:00:00', '02:00:00', '04:00:00', '06:00:00', '08:00:00', '10:00:00', '12:00:00', '14:00:00', '16:00:00'],
    'ENTRIESn': [3144312, 8936644, 3144335, 8936658, 3144353, 8936687, 3144424, 8936819, 3144594],
    'EXITSn': [1088151, 13755385,  1088159, 13755393,  1088177, 13755598, 1088231, 13756191,  1088275]

# view the first entries of a dataframe:
df.head(3)

# view the last entries of a dataframe:
df.tail(3)

# get basic info on the dataframe:
df.info()

# convert pandas dataframe to numpy array
numpy_array = df.values

# Zip the 2 lists together into one list of (key,value) tuples: zipped. (first zip together, then convert to list object)
zipped = list(zip(list_keys,list_values))

# Build a dictionary with the zipped list: data
data = dict(zipped)

# Build and inspect a DataFrame from the dictionary: df
df = pd.DataFrame(data)
print(df)

# rename the Dataframe columns with .columns method
list_labels = ['year', 'artist', 'song', 'chart weeks']
df.columns= list_labels

# Concatenate columns in new dataframe
g_column_list = list(df_g.columns.values)                   # google sheet column names in list format
frames = [df_g[g_column_list[1]], df_g[g_column_list[3]]]   # create list of columns to concatenate
df_gclean = pd.concat(frames)								# concatenate in new data frame

# Drop columns with a list
# Remove the appropriate columns: df_dropped
df_dropped = df.drop(list_to_drop,axis='columns')





#################


### MIN, MAX, MEAN

# Print the minimum value of the Engineering column
print(df['Engineering'].mean())

# Print the maximum value of the Engineering column
print(df['Engineering'].max())

# Construct the mean percentage per year: mean.  Axis argument defines mean over row(?)
mean = df.mean(axis='columns')

# Plot the average percentage per year
mean.plot()

# Display the plot
plt.show()




### QUANTILES 

# Print the number of countries reported in 2015
print(df['2015'].count())

# Print the 5th and 95th percentiles
print(df.quantile([0.05,0.95]))

# Generate a box plot
years = ['1800','1850','1900','1950','2000']
df[years].plot(kind='box')
plt.show()




### FILTERING DATA BY COLUMN

# Compute the global mean and global standard deviation: global_mean, global_std
global_mean = df.mean()
global_std = df.std()

# Filter the US population from the origin column: us
us = df.loc[df['origin'] == 'US']

# Compute the US mean and US standard deviation: us_mean, us_std
us_mean = us.mean()
us_std = us.std()

# Print the differences
print(us_mean - global_mean)
print(us_std - global_std)



### INDEXING AND SELECTING:   .iloc / .loc / .ix

See this stack overflow explanation:  http://stackoverflow.com/questions/31593201/pandas-iloc-vs-ix-vs-loc-explanation

--- .loc works on labels in the index.
--- .iloc works on the positions in the index (so it only takes integers).
--- .ix usually tries to behave like loc but falls back to behaving like iloc if the label is not in the index.
--- df[column_name] will return the values of the whole column






####################




### BASIC LINE PLOT


# Import Plotting Modules
import matplotlib.pyplot as plt
import pandas as pd

# Create a plot with color='red'
df.plot(color='red')

# Add a title
plt.title('Temperature in Austin')

# Define margin
plt.margins(0.2)

# Specify the x-axis label
plt.xlabel('Hours since midnight August 1, 2010')

# Specify the y-axis label
plt.ylabel('Temperature (degrees F)')

# Display the plot
plt.show()



##  MULTI-LINE PLOT


# DataCamp: Plotting DataFrames
# Comparing data from several columns can be very illuminating. Pandas makes doing so easy with multi-column DataFrames. By default, calling df.plot() will cause pandas to over-plot all column data, with each column as a single line. In this exercise, we have pre-loaded three columns of data from a weather data set - temperature, dew point, and pressure - but the problem is that pressure has different units of measure. The pressure data, measured in Atmospheres, has a different vertical scaling than that of the other two data columns, which are both measured in degrees Fahrenheit.
# Your job is to plot all columns as a multi-line plot, to see the nature of vertical scaling problem. Then, use a list of column names passed into the DataFrame df[column_list] to limit plotting to just one column, and then just 2 columns of data. When you are finished, you will have created 4 plots. You can cycle through them by clicking on the 'Previous Plot' and 'Next Plot' buttons.
# As in the previous exercise, inspect the DataFrame df in the IPython Shell using the .head() and .info() methods.

# Plot all columns (default)
df.plot()
plt.show()

# Plot all columns as subplots
df.plot(subplots=True)
plt.show()

# Plot just the Dew Point data
column_list1 = ["Dew Point (deg F)"]
df[column_list1].plot()
plt.show()

# Plot the Dew Point and Temperature data, but not the Pressure data
column_list2 = ['Temperature (deg F)','Dew Point (deg F)']
df[column_list2].plot()
plt.show()



##  SCATTER PLOT

# Pandas scatter plots are generated using the kind='scatter' keyword argument. Scatter plots require that the x and y columns be chosen by specifying the x and y parameters inside .plot(). Scatter plots also take an s keyword argument to provide the radius of each circle to plot in pixels.
# In this exercise, you're going to plot fuel efficiency (miles-per-gallon) versus horse-power for 392 automobiles manufactured from 1970 to 1982 from the UCI Machine Learning Repository.
# The size of each circle is provided as a NumPy array called sizes. This array contains the normalized 'weight' of each automobile in the dataset.
# All necessary modules have been imported and the DataFrame is available in the workspace as df.

# Generate a scatter plot
df.plot(kind='scatter', x='hp', y='mpg', s=sizes)

# Add the title
plt.title('Fuel efficiency vs Horse-power')

# Add the x-axis label
plt.xlabel('Horse-power')

# Add the y-axis label
plt.ylabel('Fuel efficiency (mpg)')

# Display the plot
plt.show()




##  BOX & WHISKER PLOTS

# While pandas can plot multiple columns of data in a single figure, making plots that share the same x and y axes, there are cases where two columns cannot be plotted together because their units do not match. The .plot() method can generate subplots for each column being plotted. Here, each plot will be scaled independently.
# In this exercise your job is to generate box plots for fuel efficiency (mpg) and weight from the automobiles data set. To do this in a single figure, you'll specify subplots=True inside .plot() to generate two separate plots.
# All necessary modules have been imported and the automobiles dataset is available in the workspace as df.
# Make a list of the column names to be plotted: cols

cols = ['weight','mpg']

# Generate the box plots
df[cols].plot(kind='box',subplots=True)

# Display the plot
plt.show()




# SLICING DATAFRAME AND MAKING SUBPLOTS


# Inside plt.subplots(), specify the nrows and ncols parameters so that there are 3 rows and 1 column.
# Filter the rows where the 'pclass' column has the values 1 and generate a box plot of the 'fare' column.
# Filter the rows where the 'pclass' column has the values 2 and generate a box plot of the 'fare' column.
# Filter the rows where the 'pclass' column has the values 3 and generate a box plot of the 'fare' column.

# Display the box plots on 3 separate rows and 1 column
fig, axes = plt.subplots(nrows=3,ncols=1)

# Generate a box plot of the fare prices for the First passenger class
titanic.loc[titanic['pclass'] == 1].plot(ax=axes[0], y='fare', kind='box')

# Generate a box plot of the fare prices for the Second passenger class
titanic.loc[titanic['pclass'] == 2].plot(ax=axes[1], y='fare', kind='box')

# Generate a box plot of the fare prices for the Third passenger class
titanic.loc[titanic['pclass'] == 3].plot(ax=axes[2], y='fare', kind='box')

# Display the plot
plt.show()



# PDF AND CFD PLOTS

# Construct a CDF of august_2011_high
august_2011_high.plot(kind='hist', normed=True, cumulative=True, bins=25)
# Plot the PDF
df.fraction.plot(ax=axes[0], kind='hist', normed=True, bins=30, range=(0,.3))
plt.show()






#################




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

# Good post with resample options: http://benalexkeen.com/resampling-time-series-data-with-pandas/

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



