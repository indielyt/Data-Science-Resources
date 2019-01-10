# Datacamp Intro to Portfolio Risk Management in Python

# Chapter 1: Univariate Investment Risk and Returns

	# Financial timeseries data
	# Calculating financial returns
    # Return distributions
    # First moment: Mu
    # Second moment: Variance
    # Annualizing variance
    # Third moment: Skewness
    # Fourth moment: Kurtosis

# Chapter 2: Portfolio Investing

    # Calculating portfolio returns
    # Equal weighted portfolios
    # Market-cap weighted portfolios
    # The correlation matrix
    # The co-variance matrix














######## Chapter 1: Univariate Investment Risk and Returns














### Financial timeseries data

# In finance, it is common to be working with a CSV (comma-seperate-value) 
# "flat" file of a timeseries of many different assets with their prices, returns, 
# or other data over time. Sometimes the data is stored in databases, but more 
# often than not, even large banks still use spreadsheets.

# In this exercise, you have been given a timeseries of trading data for 
# Microsoft stock as a .csv file stored at the url fpath_csv. When you finish 
# the exercise, take note of the various types of data stored in each column.

# You will be using pandas to read in the CSV data as a DataFrame.

# Import pandas as pd
import pandas as pd

# Read in the csv file and parse dates
StockPrices = pd.read_csv(fpath_csv, parse_dates=['Date'])

# Ensure the prices are sorted by Date
StockPrices = StockPrices.sort_values(by='Date')

# Print only the first five rows of StockPrices
print(StockPrices.head())

# output:
Date    Open    High     Low     Close    Volume   Adjusted
    0 2000-01-03  88.777  89.722  84.712  58.28125  53228400  38.527809
    1 2000-01-04  85.893  88.588  84.901  56.31250  54119000  37.226345
    2 2000-01-05  84.050  88.021  82.726  56.90625  64059600  37.618851
    3 2000-01-06  84.853  86.130  81.970  55.00000  54976600  36.358688
    4 2000-01-07  82.159  84.901  81.166  55.71875  62013600  36.833828







### Calculating financial returns

# The file you loaded in the previous exercise included daily Open, High, Low, Close, Adjusted Close, 
# and Volume data, often referred to as OHLCV data.  The Adjusted Close column is the most important. 
# It is normalized for stock splits, dividends, and other corporate actions, and is a true 
# reflection of the return of the stock over time. You will be using the adjusted close price 
# to calculate the returns of the stock in this exercise.  StockPrices from the previous exercise 
# is available in your workspace, and matplotlib.pyplot is imported as plt

# Calculate the daily returns of the adjusted close price
StockPrices['Returns'] = StockPrices['Adjusted'].pct_change()

# Check the first five rows of StockPrices
print(StockPrices.head())

# Plot the returns column over time
StockPrices['Returns'].plot()
plt.show()

<script.py> output:
                  Open    High     Low     Close    Volume   Adjusted   Returns
    Date                                                                       
    2000-01-03  88.777  89.722  84.712  58.28125  53228400  38.527809       NaN
    2000-01-04  85.893  88.588  84.901  56.31250  54119000  37.226345 -0.033780
    2000-01-05  84.050  88.021  82.726  56.90625  64059600  37.618851  0.010544
    2000-01-06  84.853  86.130  81.970  55.00000  54976600  36.358688 -0.033498
    2000-01-07  82.159  84.901  81.166  55.71875  62013600  36.833828  0.013068







### Return distributions

# In order to analyze the probability of outliers in returns, it is helpful to 
# visualize the historical returns of a stock using a histogram.

# You can use the histogram to show the historical density or frequency of a 
# given range of returns. Note the outliers on the left tail of the return 
# distribution are what you often want to avoid, as they represent large negative 
# daily returns. Outliers on the right hand side of the distribution are normally 
# particularly good events for the stock such as a positive earnings surprise.

# StockPrices from the previous exercise is available in your workspace, and 
# matplotlib.pyplot is imported as plt

# Convert the decimal returns into percentage returns
percent_return = StockPrices['Returns']*100

# Drop the missing values
returns_plot = percent_return.dropna()

# Plot the returns histogram
plt.hist(returns_plot, bins=75)
plt.show()







### First moment: Mu

# Note: Probability distributions have 4 'moments'

# You can calculate the average historical return of a stock by using numpy's mean() function.
# When you are calculating the average daily return of a stock, you are essentially estimating 
# the first moment ( μ ) of the historical returns distribution.
# But what use are daily return estimates to a long-term investor? You can use the formula 
# below to estimate the average annual return of a stock given the average daily return 
# and the number of trading days in a year (typically there are roughly 252 trading days in a year):

# Average Annualized Return=((1+μ)^252)−1

# The StockPrices object from the previous exercise is stored as a variable.

# Import numpy as np
import numpy as np

# Calculate the average daily return of the stock
mean_return_daily = np.mean(StockPrices['Returns'])
print(mean_return_daily)

# Calculate the implied annualized average return
mean_return_annualized = ((1+mean_return_daily)**252)-1
print(mean_return_annualized)

<script.py> output:
    0.0003777754643575774
    0.09985839482858783

# Note, returns are in decimal form, not percentage







### Second moment: Variance

# Just like you estimated the first moment of the returns distribution in the 
# last exercise, you can can also estimate the second moment, or variance of a 
# return distribution using numpy.

# In this case, you will first need to calculate the daily standard deviation 
# ( σ ), or volatility of the returns using np.std(). The variance is simply σ^2.

# StockPrices from the previous exercise is available in your workspace, and 
# numpy is imported as np.

# Calculate the standard deviation of daily return of the stock
sigma_daily = np.std(StockPrices['Returns'])
print(sigma_daily)

# Calculate the daily variance
variance_daily = sigma_daily**2
print(variance_daily)

<script.py> output:
    0.019341100408708317
    0.00037407816501973704







### Annualizing variance

# You can't annualize the variance in the same way that you annualized the mean.
# In this case, you will need to multiply σ by the square root of the number of 
# trading days in a year. There are typically 252 trading days in a calendar year. 
# Let's assume this is the case for this exercise.

# This will get you the annualized volatility, but to get annualized variance, 
# you'll need to square the annualized volatility just like you did for the daily calculation.

# sigma_daily from the previous exercise is available in your workspace, and numpy 
# is imported as np.

# Annualize the standard deviation
sigma_annualized = sigma_daily*(np.sqrt(252))
print(sigma_annualized)

# Calculate the annualized variance
variance_annualized = sigma_annualized**2
print(variance_annualized)

<script.py> output:
    0.3070304505826315    (30.7%)
    0.09426769758497373   (9.4%)







### Third moment: Skewness

# To calculate the third moment, or skewness of a returns distribution in Python, 
# you can use the skew() function from scipy.stats.

# Remember that a negative skew is a right-leaning curve, while positive skew is 
# a left-leaning curve. In finance, you would tend to want positive skewness, as 
# this would mean that the probability of large positive returns is unusually high, 
# and the negative returns are more closely clustered and predictable.

# StockPrices from the previous exercise is available in your workspace.

# Import skew from scipy.stats
from scipy.stats import skew

# Drop the missing values
clean_returns = StockPrices['Returns'].dropna()

# Calculate the third moment (skewness) of the returns distribution
returns_skewness = skew(clean_returns)
print(returns_skewness)

<script.py> output:
    0.21935459193067852

# (note, normal distribution would have skew of 0)







### Fourth moment: Kurtosis

# Finally, to calculate the fourth moment of a distribution, you can use the 
# kurtosis() function from scipy.stats.

# Note that this function actually returns the excess kurtosis, not the 4th moment 
# itself. In order to calculate kurtosis, simply add 3 to the excess kurtosis 
# returned by kurtosis().

# clean_returns from the previous exercise is available in your workspace.

# Import kurtosis from scipy.stats
from scipy.stats import kurtosis

# Calculate the excess kurtosis of the returns distribution
excess_kurtosis = kurtosis(clean_returns)
print(excess_kurtosis)

# Derive the true fourth moment of the returns distribution
fourth_moment = excess_kurtosis + 3
print(fourth_moment)

<script.py> output:
    10.31457261802553
    13.31457261802553

# note: The fourth moment (kurtosis) of the stock returns is 13.31 with an excess kurtosis 
# of 10.31. A normal distribution would tend to have a kurtosis of 3, and an excess kurtosis of 0.







### Statistical tests for normality

# In order to truly be confident in your judgement of the normality of the stock's 
# return distribution, you will want to use a true statistical test rather than 
# simply examining the kurtosis or skewness.

# You can use the shapiro() function from scipy.stats to run a Shapiro-Wilk test 
# of normality on the stock returns. The function will return two values in a list. 
# The first value is the t-stat of the test, and the second value is the p-value. 
# You can use the p-value to make a judgement about the normality of the data. If 
# the p-value is less than or equal to 0.05, you can safely reject the null hypothesis 
# of normality and assume that the data are non-normally distributed.

# clean_returns from the previous exercise is available in your workspace.

# Import shapiro from scipy.stats
from scipy.stats import shapiro

# Run the Shapiro-Wilk test on the stock returns
shapiro_results = shapiro(clean_returns)
print("Shapiro results:", shapiro_results)

# Extract the p-value from the shapiro_results
p_value = shapiro_results[1]
print("P-value: ", p_value)

<script.py> output:
    Shapiro results: (0.9003633260726929, 0.0)
    P-value:  0.0

# note: The p-value is 0, so null hypothesis of normality is rejected. The data are non-normal.















######## Chapter 2: Portfolio Investing
















### Calculating portfolio returns

# In order to build and backtest a portfolio, you have to be comfortable working 
# with the returns of multiple assets in a single object.

# In this exercise, you will be using a pandas DataFrame object, already stored as 
# the variable StockReturns, to hold the returns of multiple assets and to calculate 
# the returns of a model portfolio.

# The model portfolio is constructed with pre-defined weights for some of the 
# largest companies in the world just before January 2017:

# Company Name	Ticker	Portfolio Weight
# Apple	AAPL	12%
# Microsoft	MSFT	15%
# Exxon Mobil	XOM	8%
# Johnson & Johnson	JNJ	5%
# JP Morgan	JPM	9%
# Amazon	AMZN	10%
# General Electric	GE	11%
# Facebook	FB	14%
# AT&T	T	16%

# Note that the portfolio weights should sum to 100% in most cases

# Finish defining the portfolio weights as a numpy array
portfolio_weights = np.array([0.12, 0.15, 0.08, 0.05, 0.09, 0.10, 0.11, 0.14, 0.16])

# Calculate the weighted stock returns
WeightedReturns = StockReturns.mul(portfolio_weights, axis=1)

# Calculate the portfolio returns
StockReturns['Portfolio'] = WeightedReturns.sum(axis=1)

# Plot the cumulative portfolio returns over time
CumulativeReturns = ((1+StockReturns["Portfolio"]).cumprod()-1)
CumulativeReturns.plot()
plt.show()

see returns1.svg







### Equal weighted portfolios

# When comparing different portfolios, you often want to consider performance versus 
# a naive equally-weighted portfolio. If the portfolio doesn't outperform a simple 
# equally weighted portfolio, you might want to consider another strategy, or simply 
# opt for the naive approach if all else fails. You can expect equally-weighted 
# portfolios to tend to outperform the market when the largest companies are doing 
# poorly. This is because even tiny companies would have the same weight in your 
# equally-weighted portfolio as Apple or Amazon, for example.

# To make it easier for you to visualize the cumulative returns of portfolios, we 
# defined the function cumulative_returns_plot() in your workspace.

def cumulative_returns_plot(cols): 
    """
    cols: A list of column names to plot 
    """
    CumulativeReturns = ((1+StockReturns[cols]).cumprod()-1) 
    CumulativeReturns.plot() 
    plt.show()

# How many stocks are in your portfolio?
numstocks = 9

# Create an array of equal weights across all assets
portfolio_weights_ew = np.repeat(1/numstocks, numstocks)

# Calculate the equally-weighted portfolio returns
StockReturns['Portfolio_EW'] = StockReturns.iloc[:,0:9].mul(portfolio_weights_ew, axis=1).sum(axis=1)
cumulative_returns_plot(['Portfolio', 'Portfolio_EW'])

see CumulativeReturns.svg







### Market-cap weighted portfolios

# Conversely, when large companies are doing well, market capitalization, or 
# "market cap" weighted portfolios tend to outperform. This is because the largest 
# weights are being assigned to the largest companies, or the companies with the 
# largest market cap.

# Below is a table of the market capitalizations of the companies in your portfolio 
# just before January 2017:

# Company Name	Ticker	Market Cap ($ Billions)
# Apple	AAPL	601.51
# Microsoft	MSFT	469.25
# Exxon Mobil	XOM	349.5
# Johnson & Johnson	JNJ	310.48
# JP Morgan	JPM	299.77
# Amazon	AMZN	356.94
# General Electric	GE	268.88
# Facebook	FB	331.57
# AT&T	T	246.09

# Create an array of market capitalizations (in billions)
market_capitalizations = np.array([601.51, 469.25, 349.5, 310.48, 299.77, 356.94, 268.88, 331.57, 246.09])

# Calculate the market cap weights
mcap_weights = market_capitalizations/(np.sum(market_capitalizations))

# Calculate the market cap weighted portfolio returns
StockReturns['Portfolio_MCap'] = StockReturns.iloc[:, 0:9].mul(mcap_weights, axis=1).sum(axis=1)
cumulative_returns_plot(['Portfolio', 'Portfolio_EW', 'Portfolio_MCap'])

see marketCapitalPortfolio.svg







### The correlation matrix

# The correlation matrix can be used to estimate the linear historical relationship 
# between the returns of multiple assets. You can use the built-in .corr() method on a 
# pandas DataFrame to easily calculate the correlation matrix.

# Correlation ranges from -1 to 1. The diagonal of the correlation matrix is always 1, 
# because a stock always has a perfect correlation with itself. The matrix is symmetric, 
# which means that the lower triangle and upper triangle of the matrix are simply 
# reflections of each other since correlation is a bi-directional measurement.

# In this exercise, you will use the seaborn library to generate a heatmap.

# Calculate the correlation matrix
correlation_matrix = StockReturns.corr()

# Print the correlation matrix
print(correlation_matrix)

# Import seaborn as sns
import seaborn as sns

# Create a heatmap
sns.heatmap(correlation_matrix,
            annot=True,
            cmap="YlGnBu", 
            linewidths=0.3,
            annot_kws={"size": 8})

# Plot aesthetics
plt.xticks(rotation=90)
plt.yticks(rotation=0) 
plt.show()

see stockReturnsCorrelationMatrix.svg







### The co-variance matrix

# You can easily compute the co-variance matrix of a DataFrame of returns using 
# the .cov() method.

# The correlation matrix doesn't really tell you anything about the variance of 
# the underlying assets, only the linear relationships between assets. The 
# co-variance (a.k.a. variance-covariance) matrix, on the other hand, contains all 
# of this information, and is very useful for portfolio optimization and risk 
# management purposes.

# Calculate the covariance matrix
cov_mat = StockReturns.cov()

# Annualize the co-variance matrix
cov_mat_annual = cov_mat*252

# Print the annualized co-variance matrix
print(cov_mat_annual)

script.py> output:
              AAPL      MSFT       XOM       JNJ       JPM      AMZN        GE  \
    AAPL  0.030996  0.011400  0.001093  0.000774  0.005716  0.018805  0.000236   
    MSFT  0.011400  0.021912  0.001392  0.003899  0.004597  0.019275 -0.001315   
    XOM   0.001093  0.001392  0.012500  0.001548  0.005554  0.000354  0.004295   
    JNJ   0.000774  0.003899  0.001548  0.013092  0.001307  0.001266  0.001540   
    JPM   0.005716  0.004597  0.005554  0.001307  0.026371  0.000474  0.008283   
    AMZN  0.018805  0.019275  0.000354  0.001266  0.000474  0.043954 -0.003830   
    GE    0.000236 -0.001315  0.004295  0.001540  0.008283 -0.003830  0.039270   
    FB    0.016243  0.013682 -0.000890  0.001545  0.002631  0.023290 -0.000821   
    T     0.000152 -0.000530  0.003751  0.001780  0.006972 -0.000638  0.009849   
    
                FB         T  
    AAPL  0.016243  0.000152  
    MSFT  0.013682 -0.000530  
    XOM  -0.000890  0.003751  
    JNJ   0.001545  0.001780  
    JPM   0.002631  0.006972  
    AMZN  0.023290 -0.000638  
    GE   -0.000821  0.009849  
    FB    0.028937 -0.000708  
    T    -0.000708  0.028833






### Portfolio standard deviation

# In order to calculate portfolio volatility, you will need the covariance matrix, 
# the portfolio weights, and knowledge of the transpose operation. The transpose of 
# a numpy array can be calculated using the .T attribute. The np.dot() function is 
# the dot-product of two arrays.

# The formula for portfolio volatility is:

# σPortfolio=(wT⋅Σ⋅w)^0.5

# where:
#     σPortfolio: Portfolio volatility
#     Σ: Covariance matrix of returns
#     w: Portfolio weights (wT is transposed portfolio weights)
#     ⋅ The dot-multiplication operator

# portfolio_weights and cov_mat_annual are available in your workspace.

