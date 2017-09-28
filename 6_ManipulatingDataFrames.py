### DATACAMP - MANIPULATING DATA FRAMES. 

# Import Modules
import matplotlib.pyplot as plt
import pandas as pd

### SECTION 1: Extracting and Transforming Data

		# SHAPE OF DATAFRAME
		# POSITIONAL AND LABELED INDEXING
		# INDEXING AND COLUMN REARRANGEMENT
		# SLICING ROWS
		# SLICING COLUMNS
		# SUBSELECTING DATAFRAMES WITH LISTS
		# THRESHOLDING DATA
		# FILTERING COLUMNS USING OTHER COLUMNS
		# DROP DATA WITH NAN VALUES
		# APPLY A FUNCTION TO DF COLUMNS IN AN ELEMENT-WISE FASHION WITH .APPLY()
		# USE NUMPY'S VECTOR OPERATIONS TO TRANSFORM DATAFRAME VALUES
		# TRANSFORM VALUES BASED ON DICTIONARY LOOKUP

### SECTION 2: Advanced Indexing

		# CHANGING INDEX OF DATAFRAME
		# CHANGING INDEX LABEL NAME
		# SETTING AND SORTING A MULTI-INDEX
		# INDEXING MULTIPLE LEVELS OF A MULTIINDEX

### SECTION 3: Rearranging and Reshaping Dataframes

		# PIVOTING A SINGLE VARIABLE
		# PIVOT ALL VARIABLES
		# STACKING AND UNSTACKING
		# RESTORING INDEX ORDER
		# MELT
		# OBTAIN KEY-VALUE PAIRS USING MELT
		# USING OTHER AGGREGATIONS IN PIVOT TABLES
		# USING MARGINS IN PIVOT TABLES

### SECTION 4: Grouping Data

        # GROUPING BY MULTIPLE COLUMNS
        # GROUP BY ANOTHER SERIES
        # COMPUTING MULTIPLE AGGREGATES OF MULTIPLE COLUMNS
        # AGGREGATING ON INDEX LEVEL/FIELDS
        # GROUPING ON A FUNCTION OF THE INDEX
        # DETECTING OUTLIERS WITH Z-SCORES - USING .groupby(), .transform(), and boolean series
        # FILLING MISSING DATA (IMPUTATION) BY GROUP
        # USING APPLY AND FUNCTIONS
        # GROUPING AND FILTERING WITH APPLY
        # GROUPING AND FILTERING WITH .FILTER()
        # FILTERING AND GROUPING WITH .MAP()

### SECTION 5: Summary

		# USING .VALUE_COUNTS() FOR RANKING
		# USING .VALUE_COUNTS() FOR RANKING
		# USING .PIVOT_TABLE() TO COUNT MEDALS BY TYPE
		# APPLYING .DROP_DUPLICATES()
		# FINDING POSSIBLE ERRORS WITH .GROUPBY()
		# LOCATING SUSPICIOUS DATA
		# USING .NUNIQUE() TO RANK BY DISTINCT SPORTS
		# COUNTING USA VS USSR COLD WAR OLYMPIC SPORTS
		# COUNTING USA VS USSR COLD WAR MEDALS
		# VISUALIZING USA MEDAL COUNTS BY EDITION
		# VISUALIZING USA MEDAL COUNTS BY EDITION; AREA PLOT WITH ORDERED MEDALS



############################

### SHAPE OF DATAFRAME
print (df.shape)


###  POSITIONAL AND LABELED INDEXING

# Assign the row position of election.loc['Bedford']: x
x = 4
# Assign the column position of election['winner']: y
y = 4
# Print the boolean equivalence
print(election.iloc[x, y] == election.loc['Bedford', 'winner'])



###  INDEXING AND COLUMN REARRANGEMENT

# Create a separate dataframe with the columns reordered['winner', 'total', 'voters']: results
results = election[['winner', 'total', 'voters']]



###  SLICING ROWS

# Slice the row labels 'Perry' to 'Potter': p_counties
p_counties = election.loc['Perry':'Potter',:]

# Slice the row labels 'Potter' to 'Perry' in reverse order: p_counties_rev
p_counties_rev = election.loc['Potter':'Perry':-1,:]



###  SLICING COLUMNS

# Slice the columns from the starting column to 'Obama': left_columns
left_columns = election.loc[:,:'Obama']

# Slice the columns from 'Obama' to 'winner': middle_columns
middle_columns = election.loc[:,'Obama':'winner']

# Slice the columns from 'Romney' to the end: 'right_columns'
right_columns = election.loc[:,'Romney':]



###  SUBSELECTING DATAFRAMES WITH LISTS

# Create the list of row labels: rows
rows = ['Philadelphia', 'Centre', 'Fulton']

# Create the list of column labels: cols
cols = ['winner','Obama','Romney']

# Create the new DataFrame: three_counties
three_counties = election.loc[rows,cols]



###  THRESHOLDING DATA

# Create the boolean array: high_turnout
high_turnout = election['turnout']>70

# Filter the election DataFrame with the high_turnout array: high_turnout_df
high_turnout_df = election[high_turnout]



### FILTERING COLUMNS USING OTHER COLUMNS

# Import numpy
import numpy as np

# Create the boolean array: too_close ('margin' refers to a percentage in this case)
too_close = election['margin'] < 1

# Assign np.nan to the 'winner' column where the results were too close to call
election.loc[too_close, 'winner'] = np.nan



### DROP DATA WITH NAN VALUES

# Drop rows in df with how='any' and print the shape. 'any' drops rows where any of the column values are NaN
df.dropna(how='any').shape

# Drop rows in df with how='all' and print the shape. 'all' drops rows where all of the column values are NaN
df.dropna(how='all').shape

# Call .dropna() with thresh=1000 and axis='columns' and print the output of .info() from titanic
# Drops columns where more than 1000 of the values are missing, along axis 'columns'
print(titanic.dropna(thresh=1000, axis='columns').info())




### APPLY A FUNCTION TO DF COLUMNS IN AN ELEMENT-WISE FASHION WITH .APPLY()

# Write a function to convert degrees Fahrenheit to degrees Celsius: to_celsius
def to_celsius(F):
    return 5/9*(F - 32)

# Apply the function over 'Mean TemperatureF' and 'Mean Dew PointF': df_celsius
df_celsius = weather[['Mean TemperatureF', 'Mean Dew PointF']].apply(to_celsius)

# Reassign the columns df_celsius
df_celsius.columns = ['Mean TemperatureC', 'Mean Dew PointC']

# Print the output of df_celsius.head()
print(df_celsius.head())




###  USE NUMPY'S VECTOR OPERATIONS TO TRANSFORM DATAFRAME VALUES

# Import zscore from scipy.stats (zscore is the deviation from the mean in fractions of std deviation)
from scipy.stats import zscore

# Call zscore with election['turnout'] as input: turnout_zscore
turnout_zscore = zscore(election['turnout'])

# Print the type of turnout_zscore
print(type(turnout_zscore))

# Assign turnout_zscore to a new column: election['turnout_zscore']
election['turnout_zscore'] = turnout_zscore

# Print the output of election.head()
print(election.head())

### TRANSFORM VALUES BASED ON DICTIONARY LOOKUP

# Create the dictionary: red_vs_blue
red_vs_blue = {'Obama':'blue', 'Romney':'red'}

# Use the dictionary to map the 'winner' column to the new column: election['color']
election['color'] = election['winner'].map(red_vs_blue)

# Print the output of election.head()
print(election.head())




###########################

### CHANGING INDEX OF DATAFRAME

# Create the list of new indexes: new_idx
new_idx = [str.upper(i) for i in sales.index]

# Assign new_idx to sales.index
sales.index = new_idx

# Print the sales DataFrame
print(sales)



### CHANGING INDEX LABEL NAME

# Assign the string 'MONTHS' to sales.index.name
sales.index.name='MONTHS'

# Print the sales DataFrame
print(sales)

# Assign the string 'PRODUCTS' to sales.columns.name 
sales.columns.name='PRODUCTS'

# Print the sales dataframe again
print(sales)




### SETTING AND SORTING A MULTI-INDEX

# Set the index to be the columns ['state', 'month']: sales
sales = sales.set_index(['state','month'])

# Sort the MultiIndex: sales
sales = sales.sort_index()

# Print the sales DataFrame
print(sales)




### INDEXING MULTIPLE LEVELS OF A MULTIINDEX

sales  =
             eggs  salt  spam
state month                  
CA    1        47  12.0    17
      2       110  50.0    31
NY    1       221  89.0    72
      2        77  87.0    20
TX    1       132   NaN    52
      2       205  60.0    55
      
# Look up data for NY in month 1: NY_month1
NY_month1 = sales.loc[('NY',1)]

# Look up data for CA and TX in month 2: CA_TX_month2
# CA_TX_month2 = sales.loc[((slice('CA','TX'),slice(2))]
NY_month1 = sales.loc[(['CA','TX'],2),:]

# Look up data for all states in month 2: all_month2
all_month2 = sales.loc[(slice(None), 2),:]



#############################





## PIVOTING A SINGLE VARIABLE  (requires unique index/value pairs througout table)

# Pivot the users DataFrame: visitors_pivot
visitors_pivot = users.pivot(index='weekday',columns='city',values='visitors')




### PIVOT ALL VARIABLES  (requires unique index/value pairs througout table)

# Pivot users pivoted by both signups and visitors: pivot (Note, no 'values' attribute was names, so 
# all varialbles are pivoted and a heirarchical column structure is created)
pivot = pivot = users.pivot(index='weekday',columns='city')




### STACKING AND UNSTACKING 

# Unstack users by 'weekday': byweekday
byweekday = users.unstack(level='weekday')

# Stack byweekday by 'weekday' and print it
print(byweekday.stack(level='weekday'))




# RESTORING INDEX ORDER

# Stack 'city' back into the index of bycity: newusers
newusers = bycity.stack(level='city')

# Swap the levels of the index of newusers: newusers
newusers = newusers.swaplevel(0,1)

# Print newusers and verify that the index is not sorted
print(newusers)

# Sort the index of newusers: newusers
newusers = newusers.sort_index()

# Print newusers and verify that the index is now sorted
print(newusers)

# Verify that the new DataFrame is equal to the original
print(newusers.equals(users))




### MELT

# Reset the index: visitors_by_city_weekday
visitors_by_city_weekday = visitors_by_city_weekday.reset_index()

# Melt visitors_by_city_weekday: visitors
visitors = pd.melt(visitors_by_city_weekday, id_vars=['weekday'], value_name='visitors')




### OBTAIN KEY-VALUE PAIRS USING MELT

# Set the new index: users_idx
users_idx = users.set_index(['city','weekday'])

# Obtain the key-value pairs: kv_pairs
kv_pairs = pd.melt(users_idx,col_level=0)




###  USING OTHER AGGREGATIONS IN PIVOT TABLES

# Use a pivot table to display the count of each column: count_by_weekday1
count_by_weekday1 = users.pivot_table(index='weekday',aggfunc='count')




###  USING MARGINS IN PIVOT TABLES

# Add in the margins: signups_and_visitors_total (margins=True produces a summation row
# at the bottom of each column)
signups_and_visitors_total = users.pivot_table(index='weekday',aggfunc=sum,margins=True)





#######################


### GROUPING BY MULTIPLE COLUMNS

# Group titanic by 'pclass'
by_class = titanic.groupby('pclass')

# Aggregate 'survived' column of by_class by count
count_by_class = by_class['survived'].count()

# Group titanic by 'embarked' and 'pclass'
by_mult = titanic.groupby(['embarked', 'pclass'])

# Aggregate 'survived' column of by_mult by count
count_mult = by_mult['survived'].count()




### GROUP BY ANOTHER SERIES (both series have same index ('country'), so we
### can use the one index to group the other dataframe's values

# Read life_fname into a DataFrame: life  (data about life expectancy per country)
life = pd.read_csv(life_fname, index_col='Country')

# Read regions_fname into a DataFrame: regions  (relationships between country and region)
regions = pd.read_csv(regions_fname,index_col='Country')

# Group life by regions['region']: life_by_region
life_by_region = life.groupby(regions['region'])

# Print the mean over the '2010' column of life_by_region
print(life_by_region['2010'].mean())




# COMPUTING MULTIPLE AGGREGATES OF MULTIPLE COLUMNS

# Group titanic by 'pclass': by_class  (basically sorts)
by_class = titanic.groupby('pclass')

# Select 'age' and 'fare'  (disregarding all other columns of data about each passenger)
by_class_sub = by_class[['age','fare']]

# Aggregate by_class_sub by 'max' and 'median': aggregated (creates multi-index columns where
# age and fare columns have subcolumns of max and median values.  The rows have already been grouped by
# the pclass index)
aggregated = by_class_sub.agg(['max','median'])

# Print the maximum age in each class
print(aggregated.loc[:, ('age','max')])

# Print the median fare in each class
print(aggregated.loc[:,('fare','median')])





### AGGREGATING ON INDEX LEVEL/FIELDS

# Read the CSV file into a DataFrame and sort the index: gapminder (creates multilevel index)
gapminder = pd.read_csv('gapminder.csv',index_col=['Year','region','Country']).sort_index()

# Group gapminder by 'Year' and 'region': by_year_region
by_year_region = gapminder.groupby(level=['Year','region'])

# Define the function to compute spread: spread
def spread(series):
    return series.max() - series.min()

# Create the dictionary: aggregator
aggregator = {'population':'sum', 'child_mortality':'mean', 'gdp':spread}

# Aggregate by_year_region using the dictionary: aggregated
aggregated = by_year_region.agg(aggregator)





### GROUPING ON A FUNCTION OF THE INDEX

 # Read file: sales
sales = pd.read_csv('sales.csv',index_col='Date', parse_dates=True)

# Create a groupby object: by_day (see strftime.org for documentation)
by_day = sales.groupby(sales.index.strftime('%a'))

# Create sum: units_sum (sums sales by day of the week)
units_sum = by_day['Units'].sum()




### DETECTING OUTLIERS WITH Z-SCORES - USING .groupby(), .transform(), and boolean series

# Import zscore
from scipy.stats import zscore

# Group gapminder_2010: standardized (data is first grouped by region, then the life and 
# fertility data is standardized using zscore by region rather than whole dataset.  zscore
# is measure of number of standard deviations from the mean and is a common normalization method)
standardized = gapminder_2010.groupby('region')['life','fertility'].transform(zscore)

# Construct a Boolean Series to identify outliers: outliers (identifies which rows satisfy the 
# criteria. Note, boolean series has same order as gapminder dataframe, so we can use it later)
outliers = (standardized['life'] < -3) | (standardized['fertility'] > 3)

# Filter gapminder_2010 by the outliers: gm_outliers (.loc here used boolean series where value = 1)
gm_outliers = gapminder_2010.loc[outliers]

# Print gm_outliers
print(gm_outliers)





### FILLING MISSING DATA (IMPUTATION) BY GROUP

# Create a groupby object: by_sex_class
by_sex_class = titanic.groupby(['sex','pclass'])

# Write a function that imputes median
def impute_median(series):
    return series.fillna(series.median())

# Impute age and assign to titanic['age']
titanic.age = impute_median(by_sex_class.age.transform(impute_median))

# Print the output of titanic.tail(10)
print(titanic.tail(10))





### USING APPLY AND FUNCTIONS

# Group gapminder_2010 by 'region': regional
regional = gapminder_2010.groupby('region')

def disparity(gr):
    # Compute the spread of gr['gdp']: s
    s = gr['gdp'].max() - gr['gdp'].min()
    # Compute the z-score of gr['gdp'] as (gr['gdp']-gr['gdp'].mean())/gr['gdp'].std(): z
    z = (gr['gdp'] - gr['gdp'].mean())/gr['gdp'].std()
    # Return a DataFrame with the inputs {'z(gdp)':z, 'regional spread(gdp)':s}
    return pd.DataFrame({'z(gdp)':z , 'regional spread(gdp)':s})

# Apply the disparity function on regional: reg_disp
reg_disp = regional.apply(disparity)

# Print the disparity of 'United States', 'United Kingdom', and 'China'
print(reg_disp.loc[['United States','United Kingdom','China']])
# print(reg_disp.index.values)




### GROUPING AND FILTERING WITH APPLY

# Create a groupby object using titanic over the 'sex' column: by_sex
by_sex = titanic.groupby('sex')

def c_deck_survival(gr):
    c_passengers = gr['cabin'].str.startswith('C').fillna(False)
    return gr.loc[c_passengers, 'survived'].mean()

# Call by_sex.apply with the function c_deck_survival and print the result
c_surv_by_sex = by_sex.apply(c_deck_survival)

# Print the survival rates
print(c_surv_by_sex)




### GROUPING AND FILTERING WITH .FILTER()

# Read the CSV file into a DataFrame: sales
sales = pd.read_csv('sales.csv', index_col='Date', parse_dates=True)

# Group sales by 'Company': by_company
by_company = sales.groupby('Company')

# Compute the sum of the 'Units' of by_company: by_com_sum
by_com_sum = by_company['Units'].sum()
print(by_com_sum)

# Filter 'Units' where the sum is > 35: by_com_filt
by_com_filt = by_company.filter(lambda g:g['Units'].sum()>35)
print(by_com_filt)




### FILTERING AND GROUPING WITH .MAP()

# Create the Boolean Series: under10
under10 = (titanic['age']<10).map({True:'under 10',False:'over 10'})

# Group by under10 and compute the survival rate
survived_mean_1 = titanic.groupby(under10)['survived'].mean()
print(survived_mean_1)

# Group by under10 and pclass and compute the survival rate
survived_mean_2 = titanic.groupby([under10,'pclass'])['survived'].mean()
print(survived_mean_2)




############################

# Select the 'NOC' column of medals: country_names
country_names = medals['NOC']

# Count the number of medals won by each country: medal_counts
medal_counts = country_names.value_counts()

# Print top 15 countries ranked by medals
print(medal_counts.head(15))




### USING .PIVOT_TABLE() TO COUNT MEDALS BY TYPE

# Construct the pivot table: counted
counted = medals.pivot_table(index='NOC',columns='Medal',values='Athlete',aggfunc='count')

# Create the new column: counted['totals']
counted['totals'] = counted.sum(axis='columns')

# Sort counted by the 'totals' column
counted = counted.sort_values('totals', ascending=False)

# Print the top 15 rows of counted
print(counted.head(15))




###  APPLYING .DROP_DUPLICATES()

# Select columns: ev_gen
ev_gen = medals[['Event_gender','Gender']]

# Drop duplicate pairs: ev_gen_uniques
ev_gen_uniques = ev_gen.drop_duplicates()

# Print ev_gen_uniques
print(ev_gen_uniques)



### FINDING POSSIBLE ERRORS WITH .GROUPBY()

# Group medals by the two columns: medals_by_gender
medals_by_gender = medals.groupby(['Event_gender','Gender'])

# Create a DataFrame with a group count: medal_count_by_gender
medal_count_by_gender = medals_by_gender.count()

# Print medal_count_by_gender
print(medal_count_by_gender)




### LOCATING SUSPICIOUS DATA

# Create the Boolean Series: sus
sus = ((medals.Event_gender=='W') & (medals.Gender=='Men'))

# Create a DataFrame with the suspicious row: suspect
suspect = medals.loc[sus]

# Print suspect
print(suspect)





###  USING .NUNIQUE() TO RANK BY DISTINCT SPORTS

# Group medals by 'NOC': country_grouped
country_grouped = medals.groupby('NOC')

# Compute the number of distinct sports in which each country won medals: Nsports
Nsports = country_grouped['Sport'].nunique()

# Sort the values of Nsports in descending order
Nsports = Nsports.sort_values(ascending=False)

# Print the top 15 rows of Nsports
print(Nsports.head(15))





### COUNTING USA VS USSR COLD WAR OLYMPIC SPORTS

# Extract all rows for which the 'Edition' is between 1952 & 1988: during_cold_war
during_cold_war = ((medals['Edition']>=1952) & (medals['Edition']<=1988))

# Extract rows for which 'NOC' is either 'USA' or 'URS': is_usa_urs
is_usa_urs = medals.NOC.isin(['USA','URS'])

# Use during_cold_war and is_usa_urs to create the DataFrame: cold_war_medals
cold_war_medals = medals.loc[during_cold_war & is_usa_urs]

# Group cold_war_medals by 'NOC'
country_grouped = cold_war_medals.groupby('NOC')

# Create Nsports
Nsports = country_grouped['Sport'].nunique().sort_values(ascending=False)





### COUNTING USA VS USSR COLD WAR MEDALS

# Create the pivot table: medals_won_by_country
medals_won_by_country = medals.pivot_table(index='Edition',columns='NOC',values='Athlete',aggfunc='count')

# Slice medals_won_by_country: cold_war_usa_usr_medals
cold_war_usa_usr_medals = medals_won_by_country.loc[1952:1988, ['USA','URS']]

# Create most_medals 
most_medals = cold_war_usa_usr_medals.idxmax(axis='columns')

# Print most_medals.value_counts()
print(most_medals.value_counts())




### VISUALIZING USA MEDAL COUNTS BY EDITION

# Create the DataFrame: usa
usa = medals[medals.NOC == 'USA']

# Group usa by ['Edition', 'Medal'] and aggregate over 'Athlete'
usa_medals_by_year = usa.groupby(['Edition', 'Medal'])['Athlete'].count()

# Reshape usa_medals_by_year by unstacking
usa_medals_by_year = usa_medals_by_year.unstack(level='Medal')

# Plot the DataFrame usa_medals_by_year
usa_medals_by_year.plot()
plt.show()

# Create an area plot of usa_medals_by_year
usa_medals_by_year.plot.area()
plt.show()





### VISUALIZING USA MEDAL COUNTS BY EDITION; AREA PLOT WITH ORDERED MEDALS

# Redefine 'Medal' as an ordered categorical
medals.Medal = pd.Categorical(values=medals.Medal, categories=['Bronze','Silver','Gold'], ordered=True)

# Create the DataFrame: usa
usa = medals[medals.NOC == 'USA']

# Group usa by 'Edition', 'Medal', and 'Athlete'
usa_medals_by_year = usa.groupby(['Edition', 'Medal'])['Athlete'].count()

# Reshape usa_medals_by_year by unstacking
usa_medals_by_year = usa_medals_by_year.unstack(level='Medal')

# Create an area plot of usa_medals_by_year
usa_medals_by_year.plot.area()
plt.show()


