### DATACAMP - STATISTICAL THINKING IN PYTHON I


# Import Plotting Modules
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

# Set default plotting - Seaborn style
import seaborn as sns
sns.set()


### INTRO STUFF
		
		# CALCULATING PERCENTILES WITH NUMPY
		# Histogram
		# Defining the ECDF
		# Pearson's Correlation Coefficient

### The Normal Distribution, Binomial and Poisson Distributions

		# THE NORMAL DISTRIBUTION - PROBABILITY DENSITY FUNCTION
		# THE NORMAL DISTRIBUTION - CUMULATIVE DISTRUBUTION FUNCTION
		# INVESTIGATING DATA FOR BEING NORMALLY DISTRIBUTED
		# PROBABILITY FROM A SAMPLE SET
        # BERNOULLI TRIAL
        # BINOMIAL DISTRIBUTION FUNCTION (Same as above, but far more computationally efficient)
        # POISSON DISTRIBUTION (High number of trials(n), low probability of success(p))


### THINKING PROBABILISTICALLY - DISCRETE VARIABLES

        # GENERATING RANDOM NUMBERS USING THE NP.RANDOM MODULE
        # THE NP.RANDOM MODULE AND BERNOULLI TRIALS (random coin flips)
        # HOW MANY DEFAULTS MIGHT WE EXPECT? (repeating bernoulli trials)
        # SAMPLING OUT OF THE BINOMIAL DISTRIBUTION
        # PLOTTING THE BINOMIAL PMF 
        # RELATIONSHIP BETWEEN BINOMIAL AND POISSON DISTRIBUTIONS  (binomial approaches poisson with low probability)
        # WAS 2015 ANOMALOUS? (sampling from the poisson distribution)

### THINKING PROBABILISTICALLY - CONTINOUS VARIABLES

		#  THE NORMAL PDF
		#  THE NORMAL CDF
		#  ARE THE BELMONT STAKES RESULTS NORMALLY DISTRIBUTED? (investigating distribution of dataset)
		#  WHAT ARE THE CHANCES OF A HORSE MATCHING OR BEATING SECRETARIAT'S RECORD? (assuming normal distribution)
		#  IF YOU HAVE A STORY, YOU CAN SIMULATE IT! (exponential distributions-rare events)
		#  DISTRIBUTION OF NO-HITTERS AND CYCLES (utilizing successive poisson function)



#######################



###  CALCULATING PERCENTILES WITH NUMPY

# Specify array of percentiles: percentiles
percentiles = np.array([2.5,25,50,74,97.5])

# Compute percentiles: ptiles_vers
ptiles_vers = np.percentile(versicolor_petal_length,percentiles)
print (ptiles_vers)




# Create histogram (automatic)
_ = plt.hist(data['petal length (cm)'])
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('count')

# Create histogram (control bin number by square root of sample size, converted to integer)
n_data = len(versicolor)
n_bins = np.sqrt(n_data)
n_bins = int(n_bins)
_ = plt.hist(versicolor['petal length (cm)'],bins = n_bins)
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('count')

# Show histogram
plt.show()



### Defining the ECDF

def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""

    # Number of data points: n, convert to float for python 2
    n = float(len(data))

    # x-data for the ECDF: x sorted from low to high
    x = np.sort(data)

    # y-data for the ECDF: y as a list of ranks, low to high
    y = np.arange(1,n+1) / n
  
    return x, y

# ecdf(versicolor['petal length (cm)'])

# Compute ECDF for versicolor data: x_vers, y_vers
x_vers,y_vers = ecdf(versicolor['petal length (cm)'])
print (x_vers,y_vers)

# Generate plot
_ = plt.plot(x_vers, y_vers, marker='.', linestyle='none')

# Make the margins nice
_ = plt.margins(0.2)


# Label the axes
_ = plt.xlabel('versicolor petal length (cm)')
_ = plt.ylabel('cumulative probability')


# Display the plot
plt.show()

print ("finished")



### Pearson's correlation coefficient

def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    # Compute correlation matrix: corr_mat
    corr_mat = np.corrcoef(x, y)

    # Return entry [0,1]
    return corr_mat[0,1]

# Compute Pearson correlation coefficient for I. versicolor
r = pearson_r(versicolor_petal_width, versicolor_petal_length)

# Print the result
print(r)



#######################





###  THE NORMAL DISTRIBUTION - PROBABILITY DENSITY FUNCTION

# In this exercise, you will explore the Normal PDF and also 
# learn a way to plot a PDF of a known distribution using hacker 
# statistics. Specifically, you will plot a Normal PDF for various
#  values of the variance.

# Draw 100000 samples from Normal distribution with stds of interest: samples_std1, samples_std3, samples_std10
samples_std1 = np.random.normal(20,1,size=100000)
samples_std3 = np.random.normal(20,3,size=100000)
samples_std10 = np.random.normal(20,10,size=100000)

# Make histograms - histtype = 'step' produces a smooth line plot
plt.hist(samples_std1, normed=True, histtype='step',bins=100)
plt.hist(samples_std3, normed=True, histtype='step',bins=100)
plt.hist(samples_std10, normed=True, histtype='step',bins=100)

# Make a legend, set limits and show plot
_ = plt.legend(('std = 1', 'std = 3', 'std = 10'))
plt.ylim(-0.01, 0.42)
plt.show()





###  THE NORMAL DISTRIBUTION - CUMULATIVE DISTRUBUTION FUNCTION

def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n, convert to float for python 2
    n = float(len(data))
    # x-data for the ECDF: x sorted from low to high
    x = np.sort(data)
    # y-data for the ECDF: y as a list of ranks, low to high
    y = np.arange(1,n+1) / n
    return x, y

# Generate CDFs
x_std1, y_std1 = ecdf(samples_std1)
x_std3, y_std3 = ecdf(samples_std3)
x_std10, y_std10 = ecdf(samples_std10)

# Plot CDFs
_ = plt.plot(x_std1, y_std1, marker='.', linestyle='none')
_ = plt.plot(x_std3, y_std3, marker='.', linestyle='none')
_ = plt.plot(x_std10, y_std10, marker='.', linestyle='none')

# Make 2% margin
plt.margins(0.02)

# Make a legend and show the plot
_ = plt.legend(('std = 1', 'std = 3', 'std = 10'), loc='lower right')
plt.show()





###  INVESTIGATING DATA FOR BEING NORMALLY DISTRIBUTED

# Since 1926, the Belmont Stakes is a 1.5 mile-long race of 
# 3-year old thoroughbred horses. Secretariat ran the fastest 
# Belmont Stakes in history in 1973. While that was the fastest 
# year, 1970 was the slowest because of unusually wet and 
# sloppy conditions. With these two outliers removed from the 
# data set, compute the mean and standard deviation of the Belmont 
# winners' times. Sample out of a Normal distribution with this mean 
# and standard deviation using the np.random.normal() function and 
# plot a CDF. Overlay the ECDF from the winning Belmont times. Are 
# these close to Normally distributed?

# Note: Justin scraped the data concerning the Belmont Stakes from 
# the Belmont Wikipedia page.

# Compute mean and standard deviation: mu, sigma
mu = np.mean(belmont_no_outliers)
sigma = np.std(belmont_no_outliers)

# Sample out of a normal distribution with this mu and sigma: samples
samples = np.random.normal(mu,sigma,size=10000)

# Get the CDF of the samples and of the data
x_theor, y_theor = ecdf(samples)
x, y = ecdf(belmont_no_outliers)

# Plot the CDFs and show the plot
_ = plt.plot(x_theor, y_theor)
_ = plt.plot(x, y, marker='.', linestyle='none')
plt.margins(0.02)
_ = plt.xlabel('Belmont winning time (sec.)')
_ = plt.ylabel('CDF')
plt.show()





###  PROBABILITY FROM A SAMPLE SET

# Assume that the Belmont winners' times are Normally distributed 
# (with the 1970 and 1973 years removed), what is the probability 
# that the winner of a given Belmont Stakes will run it as fast 
# or faster than Secretariat?

# Take a million samples out of the Normal distribution: samples
samples = np.random.normal(mu,sigma,1000000)

# Compute the fraction that are faster than 144 seconds: prob
prob = np.sum(samples<=144)/len(samples)

# Print the result
print('Probability of besting Secretariat:', prob)





###  BERNOULLI TRIAL

# You can think of a Bernoulli trial as a flip of a possibly biased coin. Specifically, 
# each coin flip has a probability p of landing heads (success) and probability 1-p of 
# landing tails (failure). In this exercise, you will write a function to perform 
# n Bernoulli trials, "perform_bernoulli_trials(n, p)", which returns the number of successes 
# out of n Bernoulli trials, each of which has probability p of success. To perform each 
# Bernoulli trial, use the np.random.random() function, which returns a random number 
# between zero and one.



### Conceptually, here's what going on (use the binomial function below though for efficiency)

def perform_bernoulli_trials(n, p):
    """Perform n Bernoulli trials with success probability p
    and return number of successes."""
    # Initialize number of successes: n_success
    n_success = 0


    # Perform trials
    for i in range(n):
        # Choose random number between zero and one: random_number
        random_number = np.random.random()

        # If less than p, it's a success so add one to n_success
        if random_number < p:
            n_success += 1

    return n_success

# Use the function to predict how many mortgage failures out of 1000 loans

# Seed random number generator
np.random.seed(42)

# Initialize the number of defaults: n_defaults
n_defaults = np.empty(1000)

# Compute the number of defaults
for i in range(10):
    n_defaults[i] = perform_bernoulli_trials(100,0.05)

print (type(n_defaults))
# Plot the histogram with default number of bins; label your axes
_ = plt.hist(n_defaults, normed=True)
_ = plt.xlabel('number of defaults out of 1000 loans')
_ = plt.ylabel('probability')

# Show the plot
plt.show()




### BINOMIAL DISTRIBUTION FUNCTION (Same as above, but far more computationally efficient)

# Take 10,000 samples out of the binomial distribution: n_defaults
n_defaults = np.random.binomial(n=100,p=0.05,size=10000)
# Compute bin edges: bins
bins = np.arange(0, max(n_defaults) + 1.5) - 0.5

# Generate histogram
_ = plt.hist(n_defaults, normed = True, bins=bins)

# Set margins
_ = plt.margins(0.02)

# Label axes
_ = plt.xlabel('Number of Defaults / 100 Loans')
_ = plt.ylabel('Cumulative Distribution')

# Show the plot
plt.show()



### POISSON DISTRIBUTION (High number of trials(n), low probability of success(p))

# You just heard that the Poisson distribution is a limit of the Binomial 
# distribution for rare events. This makes sense if you think about the 
# stories. Say we do a Bernoulli trial every minute for an hour, each with 
# a success probability of 0.1. We would do 60 trials, and the number of 
# successes is Binomially distributed, and we would expect to get 
# about 6 successes. This is just like the Poisson story we discussed in the video, 
# where we get on average 6 hits on a website per hour. So, the Poisson 
# distribution with arrival rate equal to npnp approximates a Binomial 
# distribution for nn Bernoulli trials with probability pp of success 
# (with nn large and pp small). Importantly, the Poisson distribution is 
# often simpler to work with because it has only one parameter instead of 
# two for the Binomial distribution.

# Let's explore these two distributions computationally. You will compute the mean 
# and standard deviation of samples from a Poisson distribution with an arrival 
# rate of 10. Then, you will compute the mean and standard deviation of samples 
# from a Binomial distribution with parameters nn and pp such that np=10np=10.

# Draw 10,000 samples out of Poisson distribution: samples_poisson
# The "10" refers to trials times probablity of success or (n*p)
samples_poisson = np.random.poisson(10, size = 10000)

# Print the mean and standard deviation
print('Poisson:     ', np.mean(samples_poisson),
                       np.std(samples_poisson))

# Specify values of n and p to consider for Binomial: n, p
n = [20,100,1000]
p = [0.5, 0.1, 0.01]


# Draw 10,000 samples for each n,p pair: samples_binomial
for i in range(3):
    samples_binomial = np.random.binomial(n=n[i],p=p[i],size=10000)

    # Print results
    print('n =', n[i], 'Binom:', np.mean(samples_binomial),
                                 np.std(samples_binomial))







######################





###  GENERATING RANDOM NUMBERS USING THE NP.RANDOM MODULE

# Seed the random number generator
np.random.seed(42)

# Initialize random numbers: random_numbers
random_numbers = np.empty(100000)

# Generate random numbers by looping over range(100000)
for i in range(100000):
    random_numbers[i] = np.random.random()

# This is the fastest way to create and array of random numbers
random_numbers2 = np.random.random(size=10)

# Plot a histogram
_ = plt.hist(random_numbers)

# Show the plot
plt.show()





###  THE NP.RANDOM MODULE AND BERNOULLI TRIALS (random coin flips)

def perform_bernoulli_trials(n, p):
    """Perform n Bernoulli trials with success probability p
    and return number of successes."""
    # Initialize number of successes: n_success
    n_success = 0


    # Perform trials
    for i in range(n):
        # Choose random number between zero and one: random_number
        random_number = np.random.random()

        # If less than p, it's a success so add one to n_success
        if random_number < p:
            n_success += 1

    return n_success





###  HOW MANY DEFAULTS MIGHT WE EXPECT? (repeating bernoulli trials)

# Seed random number generator
np.random.seed(42)

# Initialize the number of defaults: n_defaults
n_defaults = np.empty(1000)

# Compute the number of defaults
for i in range(1000):
    n_defaults[i] = perform_bernoulli_trials(100,0.05)

# Plot the histogram with default number of bins; label your axes
_ = plt.hist(n_defaults, normed=True)
_ = plt.xlabel('number of defaults out of 100 loans')
_ = plt.ylabel('probability')

# Show the plot
plt.show()





###  SAMPLING OUT OF THE BINOMIAL DISTRIBUTION

# Take 10,000 samples out of the binomial distribution: n_defaults
n_defaults = np.random.binomial(n=100,p=0.05,size=10000)

# Compute CDF: x, y  (ecdf is a custom function, see section 1)
x,y = ecdf(n_defaults)

# Plot the CDF with axis labels
_ = plt.plot(x, y, marker='.', linestyle='none')
_ = plt.xlabel('Number of Defaults out of 100 Loans')
_ = plt.ylabel('cumulative distribution function')

# Show the plot
plt.show()





###  PLOTTING THE BINOMIAL PMF 

# We want the bins centered on the integers. So, the edges of the bins 
# should be -0.5, 0.5, 1.5, 2.5, ... up to max(n_defaults) + 1.5. You can 
# generate an array like this using np.arange() and then subtracting 0.5 from the array.

# Compute bin edges: bins
bins = np.arange(0, max(n_defaults) + 1.5) - 0.5

# Generate histogram
_ = plt.hist(n_defaults, normed = True, bins=bins)

# Set margins
_ = plt.margins(0.02)

# Label axes
_ = plt.xlabel('Number of Defaults / 100 Loans')
_ = plt.ylabel('Cumulative Distribution')

# Show the plot
plt.show()




###  RELATIONSHIP BETWEEN BINOMIAL AND POISSON DISTRIBUTIONS  (binomial approaches poisson with low probability)

# Draw 10,000 samples out of Poisson distribution: samples_poisson
samples_poisson = np.random.poisson(10, size = 10000)

# Print the mean and standard deviation
print('Poisson:     ', np.mean(samples_poisson),
                       np.std(samples_poisson))

# Specify values of n and p to consider for Binomial: n, p
n = [20,100,1000]
p = [0.5, 0.1, 0.01]

# Draw 10,000 samples for each n,p pair: samples_binomial
for i in range(3):
    samples_binomial = np.random.binomial(n=n[i],p=p[i],size=10000)

    # Print results
    print('n =', n[i], 'Binom:', np.mean(samples_binomial),
                                 np.std(samples_binomial))




###  WAS 2015 ANOMALOUS? (sampling from the poisson distribution)

# Draw 10,000 samples out of Poisson distribution: n_nohitters
n_nohitters = np.random.poisson(251/115,size=10000)

# Compute number of samples that are seven or greater: n_large
n_large = np.sum(n_nohitters>=7)

# Compute probability of getting seven or more: p_large
p_large = n_large/len(n_nohitters)

# Print the result
print('Probability of seven or more no-hitters:', p_large)






###########################






###  THE NORMAL PDF

# Draw 100000 samples from Normal distribution with stds of interest: samples_std1, samples_std3, samples_std10
# Use syntax np.random.normal(mean,std.dev,size)
samples_std1 = np.random.normal(20,1,size=100000)
samples_std3 = np.random.normal(20,3,size=100000)
samples_std10 = np.random.normal(20,10,size=100000)

# Make histograms
plt.hist(samples_std1, normed=True, histtype='step',bins=100)
plt.hist(samples_std3, normed=True, histtype='step',bins=100)
plt.hist(samples_std10, normed=True, histtype='step',bins=100)

# Make a legend, set limits and show plot
_ = plt.legend(('std = 1', 'std = 3', 'std = 10'))
plt.ylim(-0.01, 0.42)
plt.show()




###  THE NORMAL CDF

# Generate CDFs
x_std1, y_std1 = ecdf(samples_std1)
x_std3, y_std3 = ecdf(samples_std3)
x_std10, y_std10 = ecdf(samples_std10)

# Plot CDFs
_ = plt.plot(x_std1, y_std1, marker='.', linestyle='none')
_ = plt.plot(x_std3, y_std3, marker='.', linestyle='none')
_ = plt.plot(x_std10, y_std10, marker='.', linestyle='none')

# Make 2% margin
plt.margins(0.02)

# Make a legend and show the plot
_ = plt.legend(('std = 1', 'std = 3', 'std = 10'), loc='lower right')
plt.show()





### ARE THE BELMONT STAKES RESULTS NORMALLY DISTRIBUTED? (investigating distribution of dataset)

# Compute mean and standard deviation: mu, sigma
mu = np.mean(belmont_no_outliers)
sigma = np.std(belmont_no_outliers)

# Sample out of a normal distribution with this mu and sigma: samples
samples = np.random.normal(mu,sigma,size=10000)

# Get the CDF of the samples and of the data
x_theor, y_theor = ecdf(samples)
x, y = ecdf(belmont_no_outliers)

# Plot the CDFs and show the plot
_ = plt.plot(x_theor, y_theor)




###  WHAT ARE THE CHANCES OF A HORSE MATCHING OR BEATING SECRETARIAT'S RECORD? (assuming normal distribution)

# Take a million samples out of the Normal distribution: samples
samples = np.random.normal(mu,sigma,1000000)

# Compute the fraction that are faster than 144 seconds: prob
prob = np.sum(samples<=144)/len(samples)

# Print the result
print('Probability of besting Secretariat:', prob)





###  IF YOU HAVE A STORY, YOU CAN SIMULATE IT! (exponential distributions-rare events)

def successive_poisson(tau1, tau2, size=1):
    # Draw samples out of first exponential distribution: t1
    t1 = np.random.exponential(tau1, size)

    # Draw samples out of second exponential distribution: t2
    t2 = np.random.exponential(tau2, size)

    return t1 + t2



###  DISTRIBUTION OF NO-HITTERS AND CYCLES (utilizing successive poisson function)

# Now, you'll use your sampling function to compute the waiting time 
# to observer a no-hitter and hitting of the cycle. The mean waiting 
# time for a no-hitter is 764 games, and the mean waiting time for 
# hitting the cycle is 715 games.

# Draw samples of waiting times: waiting_times
waiting_times = successive_poisson(764,715,size=100000)

# Make the histogram
plt.hist(waiting_times, normed=True, histtype='step',bins=100)

# Label axes
_ = plt.xlabel('total waiting time (games)')
_ = plt.ylabel('PDF')

# Show the plot
plt.show()

