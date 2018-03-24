# Datacamp STATISTICAL THINKING IN PYTHON II




### CHAPTER 1: PARAMETER ESTIMATION BY OPTIMIZATION

        ###  HOW OFTEN DO WE GET NO HITTERS? (Sampling from Exponential Distribution)
        ###  DO THE DATA FOLLOW OUR STORY? (Comparing real vs. theoretical distributions)
        ###  HOW IS THIS PARAMETER OPTIMAL?  (Checking best fit using tau(mean) paramter)
        ###  EDA OF LITERACY/FERTILITY DATA (exploratory data analysis)
        ###  LINEAR REGRESSION  (np.polyfit())
        ###  HOW IS IT OPTIMAL?  (Determining RSS - residual sum of squares)
        ###  LINEAR REGRESSION ON APPROPRIATE ANSCOMBE DATA 


### CHAPTER 2: BOOTSTRAP CONFIDENCE INTERVALS

		### VISUALIZING BOOTSTRAP SAMPLES generating bootstrap sample, comparing to actual data)
		### GENERATING MANY BOOTSTRAP REPLICATES
		### BOOTSTRAP REPLICATES OF THE MEAN AND THE SEM(standard error of mean)
		### CONFIDENCE INTERVALS OF RAINFALL DATA (using mean of bootstrap replicates)
		### BOOTSTRAP REPLICATES OF OTHER STATISTICS (using variance of bootstrap replicates)
		### CONFIDENCE INTERVAL ON THE RATE OF NO-HITTERS (using mean again)
		### A FUNCTION TO DO PAIRS BOOTSTRAP (bootstrap sampling x,y for linear regression)
		### PAIRS BOOTSTRAP OF LITERACY/FERTILITY DATA (use the pairs bootstrap function)
		### PLOTTING BOOTSTRAP REGRESSIONS (really cool! scatter of data with boostrap sample regressions)


### CHAPTER 3: INTRODUCTION TO HYPOTHESIS TESTING

        ### GENERATING A PERMUTATION SAMPLE
        ### VISUALIZING PERMUTATION SAMPLING
        ### GENERATING PERMUTATION REPLICATES (applying a function to permutation samples to derive test statistic)
        ### LOOK BEFORE YOU LEAP: EDA BEFORE HYPOTHESIS TESTING (swarmplot for EDA)
        ### PERMUTATION TEST ON FROG DATA (using permutation sample and draw_perm_reps functions)
        ### A ONE-SAMPLE BOOTSTRAP HYPOTHESIS TEST (when we compare one full dataset to just statistics of another dataset)
        ### A BOOTSTRAP TEST FOR IDENTICAL DISTRIBUTIONS (similar to permutation test, but more versatile)
        ### A TWO SAMPLE BOOTSTRAP HYPOTHESIS TEST FOR DIFFERENCE OF MEANS


### CHAPTER 4: HYPOTHESIS TESTING EXAMPLES

		 ### THE VOTE FOR THE CIVIL RIGHTS ACT IN 1964 (using permutation sampling, replicates, and hypothesis testing)
		 ### A TIME ON WEBSITE ANALOG (comparing before and after dead ball regulations in MLB)  
		 ### HYPOTHESIS TEST ON PEARSON CORRELATION (how to test randomness of correlation by p-value) 
		 ### BOOTSTRAP HYPOTHESIS TEST ON BEE SPERM COUNTS (comparison of means by bootstrap and p-value)


### CHAPTER 5: CASE STUDY

		###  EDA OF BEAK DEPTHS OF DARWIN'S FINCHES
		###  ECDFs OF BEAK DEPTHS
		###  PARAMETER ESTIMATES OF BEAK DEPTHS (confidence intervals of different means)
		###  Hypothesis test: Are beaks deeper in 2012?
		###  EDA OF BEAK LENGTH AND DEPTH
		###  DISPLAYING THE LINEAR REGRESSION RESULTS 
		###  LINEAR REGRESSIONS
		###  BEAK LENGTH TO DEPTH RATIO
		###  EDA OF HERITABILITY
		###  Correlation of offspring and parental data
		###  Pearson correlation of offspring and parental data
		###  Measuring heritability
		###  Is beak depth heritable at all in *G. scandens*?











############################ Chapter 1: PARAMETER ESTIMATION BY OPTIMIZATION













###  HOW OFTEN DO WE GET NO HITTERS? (Sampling from Exponential Distribution)

# Seed random number generator
np.random.seed(42)

# Compute mean no-hitter time: tau
tau = np.mean(nohitter_times)

# Draw out of an exponential distribution with parameter tau: inter_nohitter_time
inter_nohitter_time = np.random.exponential(tau, 100000)

# Plot the PDF and label axes
_ = plt.hist(inter_nohitter_time,
             bins=50, normed=True, histtype='step')
_ = plt.xlabel('Games between no-hitters')
_ = plt.ylabel('PDF')

# Show the plot
plt.show()





###  DO THE DATA FOLLOW OUR STORY? (Comparing real vs. theoretical distributions)

# Create an ECDF from real data: x, y
x, y = ecdf(nohitter_times)

# Create a CDF from theoretical samples: x_theor, y_theor.  From previous exercise)
x_theor, y_theor = ecdf(inter_nohitter_time)

# Overlay the plots
plt.plot(x_theor, y_theor)
plt.plot(x, y, marker='.', linestyle='none')

# Margins and axis labels
plt.margins(0.02)
plt.xlabel('Games between no-hitters')
plt.ylabel('CDF')

# Show the plot
plt.show()




### HOW IS THIS PARAMETER OPTIMAL?  (Checking best fit using tau(mean) paramter)

# Plot the theoretical CDFs
plt.plot(x_theor, y_theor)
plt.plot(x, y, marker='.', linestyle='none')
plt.margins(0.02)
plt.xlabel('Games between no-hitters')
plt.ylabel('CDF')

# Take samples with half tau: samples_half
samples_half = np.random.exponential(tau/2,10000)

# Take samples with double tau: samples_double
samples_double = np.random.exponential(2*tau,10000)

# Generate CDFs from these samples
x_half, y_half = ecdf(samples_half)
x_double, y_double = ecdf(samples_double)

# Plot these CDFs as lines
_ = plt.plot(x_half, y_half)
_ = plt.plot(x_double, y_double)

# Show the plot
plt.show()





###  EDA OF LITERACY/FERTILITY DATA (exploratory data analysis)

def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    # Compute correlation matrix: corr_mat
    corr_mat = np.corrcoef(x, y)

    # Return entry [0,1]
    return corr_mat[0,1]

# Compute Pearson correlation coefficient for I. versicolor
r = pearson_r(versicolor_petal_width, versicolor_petal_length)

# Plot the illiteracy rate versus fertility
_ = plt.plot(illiteracy, fertility, marker='.', linestyle='none')

# Set the margins and label axes
plt.margins(0.02)
_ = plt.xlabel('percent illiterate')
_ = plt.ylabel('fertility')

# Show the plot
plt.show()

# Show the Pearson correlation coefficient
print(pearson_r(illiteracy, fertility))




###  LINEAR REGRESSION  (np.polyfit())

# Plot the illiteracy rate versus fertility
_ = plt.plot(illiteracy, fertility, marker='.', linestyle='none')
plt.margins(0.02)
_ = plt.xlabel('percent illiterate')
_ = plt.ylabel('fertility')

# Perform a linear regression using np.polyfit(): a(slope), b(intercept)
a, b = np.polyfit(illiteracy,fertility,1)

# Print the results to the screen
print('slope =', a, 'children per woman / percent illiterate')
print('intercept =', b, 'children per woman')

# Make theoretical line to plot
x = np.array([0,100])
y = m * x + b

# Add regression line to your plot
_ = plt.plot(x,y)

# Draw the plot
plt.show()




###  HOW IS IT OPTIMAL?  (Determining RSS - residual sum of squares)

# Specify slopes to consider: a_vals
a_vals = np.linspace(0,0.1,200)

# Initialize sum of square of residuals: rss
rss = np.empty_like(a_vals)

# Compute sum of square of residuals for each value of a_vals; a&b from previous exercise
for i, a in enumerate(a_vals):
    rss[i] = np.sum((fertility - a*illiteracy - b)**2)

# Plot the RSS
plt.plot(a_vals, rss, '-')
plt.xlabel('slope (children per woman / percent illiterate)')
plt.ylabel('sum of square of residuals')

plt.show()




### LINEAR REGRESSION ON APPROPRIATE ANSCOMBE DATA 

# Perform linear regression: a, b
a, b = np.polyfit(x,y,1)

# Print the slope and intercept
print(a, b)

# Generate theoretical x and y data: x_theor, y_theor
x_theor = np.array([3, 15])
y_theor = a * x_theor + b

# Plot the Anscombe data and theoretical line
_ = plt.plot(x,y, marker='.',linestyle='none')
_ = plt.plot(x_theor,y_theor)

# Label the axes
plt.xlabel('x')
plt.ylabel('y')

# Show the plot
plt.show()













############################ Chapter 2: 















### VISUALIZING BOOTSTRAP SAMPLES (generating bootstrap sample, comparing to actual data)

# Bootstrap samples are sets of randomly chosen data points 
# from an existing dataset.  Bootstrap replicates
# are the statistics computed from these new sets of sampled data

for _ in range(50):
    # Generate bootstrap sample: bs_sample
    bs_sample = np.random.choice(rainfall, size=len(rainfall))

    # Compute and plot ECDF from bootstrap sample
    x, y = ecdf(bs_sample)
    _ = plt.plot(x, y, marker='.', linestyle='none',
                 color='gray', alpha=0.1)

# Compute and plot ECDF from original data
x, y = ecdf(rainfall)
_ = plt.plot(x, y, marker='.')

# Make margins and label axes
plt.margins(0.02)
_ = plt.xlabel('yearly rainfall (mm)')
_ = plt.ylabel('ECDF')

# Show the plot
plt.show()




### GENERATING MANY BOOTSTRAP REPLICATES

# My improved (consolidated version of the course supplied function)
def draw_bs_reps(data, func, size=1):
    # Initialize array of replicates: bs_replicates
    bs_replicates = np.empty(size)
    # Populate empty array with bootstrap replicates (statistic generated
    # through user supplied function) from bootstrap sample sets
    for i in range(size):
        bs_replicates[i] = func(np.random.choice(data, len(data)))

    return bs_replicates

# # Define function for generating a single replicate
# def bootstrap_replicate_1d(data, func):
# 	# Generate bootstrap replicate of 1D data
# 	bs_sample = np.random.choice(data, len(data))
# 	return func(bs_sample)

# # Define a function for generating multiple replicates
# def draw_bs_reps(data, func, size=1):
#     """Draw bootstrap replicates."""

#     # Initialize array of replicates: bs_replicates
#     bs_replicates = np.empty(size)

#     # Generate replicates
#     for i in range(size):
#         bs_replicates[i] = bootstrap_replicate_1d(data,func)

#     return bs_replicates





### BOOTSTRAP REPLICATES OF THE MEAN AND THE SEM(standard error of mean)

# Take 10,000 bootstrap replicates of the mean: bs_replicates
bs_replicates = draw_bs_reps(rainfall,np.mean,size=10000)

# Compute and print SEM
sem = np.std(rainfall) / np.sqrt(len(rainfall))
print(sem)

# Compute and print standard deviation of bootstrap replicates
bs_std = np.std(bs_replicates)
print(bs_std)

# Make a histogram of the results
_ = plt.hist(bs_replicates, bins=50, normed=True)
_ = plt.xlabel('mean annual rainfall (mm)')
_ = plt.ylabel('PDF')

# Show the plot
plt.show()





### CONFIDENCE INTERVALS OF RAINFALL DATA (using bootstrap replicates)

# Compute the 95% confidence interval
np.percentile(bs_replicates,[2.5,97.5])





### BOOTSTRAP REPLICATES OF OTHER STATISTICS (variance)

# Generate 10,000 bootstrap replicates of the variance: bs_replicates
bs_replicates = draw_bs_reps(rainfall, np.var,size=10000)

# Put the variance in units of square centimeters
bs_replicates = bs_replicates/100

# Make a histogram of the results
_ = plt.hist(bs_replicates, bins=50, normed=True)
_ = plt.xlabel('variance of annual rainfall (sq. cm)')
_ = plt.ylabel('PDF')

# Show the plot
plt.show()




### CONFIDENCE INTERVAL ON THE RATE OF NO-HITTERS (using mean again)

# Draw bootstrap replicates of the mean no-hitter time (equal to tau): bs_replicates
bs_replicates = draw_bs_reps(nohitter_times,np.mean,size=10000)

# Compute the 95% confidence interval: conf_int
conf_int = np.percentile(bs_replicates,[2.5,97.5])

# Print the confidence interval
print('95% confidence interval =', conf_int, 'games')

# Plot the histogram of the replicates
_ = plt.hist(bs_replicates, bins=50, normed=True)
_ = plt.xlabel(r'$\tau$ (games)')
_ = plt.ylabel('PDF')

# Show the plot
plt.show()





###  A FUNCTION TO DO PAIRS BOOTSTRAP (bootstrap sampling x,y for linear regression)

def draw_bs_pairs_linreg(x, y, size=1):
    """Perform pairs bootstrap for linear regression."""

    # Set up array of indices to sample from: inds
    inds = np.arange(len(x))

    # Initialize replicates: bs_slope_reps, bs_intercept_reps
    bs_slope_reps = np.empty(size)
    bs_intercept_reps = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_inds = np.random.choice(inds,size=len(inds))
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        bs_slope_reps[i], bs_intercept_reps[i] = np.polyfit(bs_x, bs_y, 1)

    return bs_slope_reps, bs_intercept_reps






###  PAIRS BOOTSTRAP OF LITERACY/FERTILITY DATA (use the pairs bootstrap function)
 
# Generate replicates of slope and intercept using pairs bootstrap
bs_slope_reps, bs_intercept_reps = draw_bs_pairs_linreg(illiteracy,fertility,1000)

# Compute and print 95% CI for slope
print(np.percentile(bs_slope_reps, [2.5,97.5]))

# Plot the histogram
_ = plt.hist(bs_slope_reps, bins=50, normed=True)
_ = plt.xlabel('slope')
_ = plt.ylabel('PDF')
plt.show()




###  PLOTTING BOOTSTRAP REGRESSIONS (really cool! scatter of data with boostrap sample regressions)

# Generate array of x-values for bootstrap lines: x
x = np.array([0,100])

# Plot the bootstrap lines
for i in range(100):
    _ = plt.plot(x, bs_slope_reps[i]*x + bs_intercept_reps[i],
                 linewidth=0.5, alpha=0.2, color='red')

# Plot the data
_ = plt.plot(illiteracy,fertility, marker='.',linestyle='none')

# Label axes, set the margins, and show the plot
_ = plt.xlabel('illiteracy')
_ = plt.ylabel('fertility')
plt.margins(0.02)
plt.show()















############################ Chapter 3: 














### GENERATING A PERMUTATION SAMPLE

# Permutation concatenates 2 data sets, then randomly rearranges them.
# Then the scrambled array is divided back into 2 data sets, the same 
# lengths as the original data sets.  

def permutation_sample(data1, data2):
    """Generate a permutation sample from two data sets."""

    # Concatenate the data sets: data
    data = np.concatenate((data1,data2))

    # Permute the concatenated array: permuted_data
    permuted_data = np.random.permutation(data)

    # Split the permuted array into two: perm_sample_1, perm_sample_2
    perm_sample_1 = permuted_data[:len(data1)]
    perm_sample_2 = permuted_data[len(data1):]

    return perm_sample_1, perm_sample_2





### VISUALIZING PERMUTATION SAMPLING

# Here we'll compare rainfall from two months by creating 50 permutation samples,
# generating a cdf of each sample, then compare it to the original data sets by 
# plotting all cdfs on the same graph. The resulting cloud of permutation cdfs lays 
# in the middle of the cdfs of the original data, indicating that these two data
# sets are not equally distributed.

for i in range(50):
    # Generate permutation samples
    perm_sample_1, perm_sample_2 = permutation_sample(rain_july,rain_november)

    # Compute ECDFs
    x_1, y_1 = ecdf(perm_sample_1)
    x_2, y_2 = ecdf(perm_sample_2)

    # Plot ECDFs of permutation sample
    _ = plt.plot(x_1, y_1, marker='.', linestyle='none',
                 color='red', alpha=0.02)
    _ = plt.plot(x_2, y_2, marker='.', linestyle='none',
                 color='blue', alpha=0.02)

# Create and plot ECDFs from original data
x_1, y_1 = ecdf(rain_july)
x_2, y_2 = ecdf(rain_november)
_ = plt.plot(x_1, y_1, marker='.', linestyle='none', color='red')
_ = plt.plot(x_2, y_2, marker='.', linestyle='none', color='blue')

# Label axes, set margin, and show plot
plt.margins(0.02)
_ = plt.xlabel('monthly rainfall (mm)')
_ = plt.ylabel('ECDF')
plt.show()




### GENERATING PERMUTATION REPLICATES (applying a function to permutation samples to derive test statistic)

def draw_perm_reps(data_1, data_2, func, size=1):
    """Generate multiple permutation replicates."""

    # Initialize array of replicates: perm_replicates
    perm_replicates = np.empty(size)

    for i in range(size):
        # Generate permutation sample   #scrambled data sets of their original length, made up of samples of the 
        # concatentated data
        perm_sample_1, perm_sample_2 = permutation_sample(data_1,data_2)

        # Compute the test statistic
        perm_replicates[i] = func(perm_sample_1,perm_sample_2)

    return perm_replicates




### LOOK BEFORE YOU LEAP: EDA BEFORE HYPOTHESIS TESTING (swarmplot for EDA)

  # Make bee swarm plot
_ = sns.swarmplot(x='ID', y='impact_force', data=df)

# Label axes
_ = plt.xlabel('frog')
_ = plt.ylabel('impact force (N)')

# Show the plot
plt.show()




### PERMUTATION TEST ON FROG DATA (using permutation sample and draw_perm_reps functions)

def diff_of_means(data_1, data_2):
    """Difference in means of two arrays."""

    # The difference of means of data_1, data_2: diff
    diff = np.mean(data_1) - np.mean(data_2)

    return diff

# Compute difference of mean impact force from experiment: empirical_diff_means
empirical_diff_means = diff_of_means(force_a, force_b)

# Draw 10,000 permutation replicates: perm_replicates
perm_replicates = draw_perm_reps(force_a, force_b,
                                 diff_of_means, size=10000)

# Compute p-value: p
p = np.sum(perm_replicates >= empirical_diff_means) / len(perm_replicates)

# Print the result
print('p-value =', p)




### A ONE-SAMPLE BOOTSTRAP HYPOTHESIS TEST (when we compare one full dataset to just statistics of another dataset)

# Make an array of translated impact forces: translated_force_b
translated_force_b = force_b - np.mean(force_b) + 0.55

# Take bootstrap replicates of Frog B's translated impact forces: bs_replicates
bs_replicates = draw_bs_reps(translated_force_b, np.mean, 10000)

# Compute fraction of replicates that are less than the observed Frog B force: p
p = np.sum(bs_replicates <= np.mean(force_b)) / 10000

# Print the p-value
print('p = ', p)





### A BOOTSTRAP TEST FOR IDENTICAL DISTRIBUTIONS (similar to permutation test, but more versatile)

# Compute difference of mean impact force from experiment: empirical_diff_means
empirical_diff_means = diff_of_means(force_a, force_b)

# Concatenate forces: forces_concat
forces_concat = np.concatenate((force_a, force_b))

# Initialize bootstrap replicates: bs_replicates
bs_replicates = np.empty(10000)

for i in range(10000):
    # Generate bootstrap sample
    bs_sample = np.random.choice(forces_concat, size=len(forces_concat))

    # Compute replicate
    bs_replicates[i] = diff_of_means(bs_sample[:len(force_a)],
                                     bs_sample[len(force_a):])

# Compute and print p-value: p
p = np.sum(bs_replicates >= empirical_diff_means) / len(bs_replicates)
print('p-value =', p)





### A TWO SAMPLE BOOTSTRAP HYPOTHESIS TEST FOR DIFFERENCE OF MEANS

# Compute mean of all forces: mean_force
mean_force = np.mean(forces_concat)

# Generate shifted arrays
force_a_shifted = force_a - np.mean(force_a) + mean_force
force_b_shifted = force_b - np.mean(force_b) + mean_force

# Compute 10,000 bootstrap replicates from shifted arrays
bs_replicates_a = draw_bs_reps(force_a_shifted, np.mean, size=10000)
bs_replicates_b = draw_bs_reps(force_b_shifted, np.mean, size=10000)

# Get replicates of difference of means: bs_replicates
bs_replicates = bs_replicates_a - bs_replicates_b

# Compute and print p-value: p
p = (np.sum(bs_replicates > empirical_diff_means) / len(bs_replicates))
print('p-value =', p)


















############################ Chapter 4: 















###  THE VOTE FOR THE CIVIL RIGHTS ACT IN 1964 (using permutation sampling, replicates, and hypothesis testing)

# The Civil Rights Act of 1964 was one of the most important pieces of legislation ever passed in the USA. 
# Excluding "present" and "abstain" votes, 153 House Democrats and 136 Republicans voted yay. However, 91 Democrats 
# and 35 Republicans voted nay. Did party affiliation make a difference in the vote?
# To answer this question, you will evaluate the hypothesis that the party of a House member has no bearing on his 
# or her vote. You will use the fraction of Democrats voting in favor as your test statistic and evaluate the 
# probability of observing a fraction of Democrats voting in favor at least as small as the observed fraction 
# of 153/244. (That's right, at least as small as. In 1964, it was the Democrats who were less progressive on 
# civil rights issues.) To do this, permute the party labels of the House voters and then arbitrarily divide 
# them into "Democrats" and "Republicans" and compute the fraction of Democrats voting yay.

# Construct arrays of data: dems, reps (votes for and against the act, as boolean values)
dems = np.array([True] * 153 + [False] * 91) #democrats
reps = np.array([True] * 136 + [False] * 35) #republicans

def frac_yay_dems(dems, reps): #only going to use the dems to test the hypothesis
    """Compute fraction of Democrat yay votes."""
    frac = np.sum(dems) / len(dems)
    return frac

# Acquire permutation samples: perm_replicates
perm_replicates = draw_perm_reps(dems, reps, frac_yay_dems, size=10000)

# Compute and print p-value: p
p = np.sum(perm_replicates <= 153/244) / len(perm_replicates)
print('p-value =', p)   






###  A TIME ON WEBSITE ANALOG (comparing before and after dead ball regulations in MLB)

# It turns out that you already did a hypothesis test analogous to an A/B test where you 
# are interested in how much time is spent on the website before and after an ad campaign. 
# The frog tongue force (a continuous quantity like time on the website) is an analog. 
# "Before" = Frog A and "after" = Frog B. Let's practice this again with something that is 
# actually a before/after scenario.
# We return to the no-hitter data set. In 1920, Major League Baseball implemented important 
# rule changes that ended the so-called dead ball era. Importantly, the pitcher was no 
# longer allowed to spit on or scuff the ball, an activity that greatly favors pitchers. 
# In this problem you will perform an A/B test to determine if these rule changes resulted 
# in a slower rate of no-hitters (i.e., longer average time between no-hitters) using the 
# difference in mean inter-no-hitter time as your test statistic. The inter-no-hitter times 
# for the respective eras are stored in the arrays nht_dead and nht_live, where "nht" is 
# meant to stand for "no-hitter time."
# Since you will be using your draw_perm_reps() function in this exercise, it may be useful 
# to remind yourself of its call signature: draw_perm_reps(d1, d2, func, size=1) or even 
# referring back to the chapter 3 exercise in which you defined it.

# Compute the observed difference in mean inter-no-hitter times: nht_diff_obs
nht_diff_obs = diff_of_means(nht_dead, nht_live)

# Acquire 10,000 permutation replicates of difference in mean no-hitter time: perm_replicates
perm_replicates = draw_perm_reps(nht_dead, nht_live, diff_of_means, 10000)

# Compute and print the p-value: p
p = np.sum(perm_replicates <= nht_diff_obs) / len(perm_replicates)
print('p-val =',p)





### HYPOTHESIS TEST ON PEARSON CORRELATION (how to test randomness of correlation by p-value)

# The observed correlation between female illiteracy and fertility may just be by chance; 
# the fertility of a given country may actually be totally independent of its illiteracy. 
# You will test this hypothesis. To do so, permute the illiteracy values but leave the 
# fertility values fixed. This simulates the hypothesis that they are totally independent 
# of each other. For each permutation, compute the Pearson correlation coefficient and 
# assess how many of your permutation replicates have a Pearson correlation coefficient 
# greater than the observed one.
# The function pearson_r() that you wrote in the prequel to this course for computing the 
# Pearson correlation coefficient is already in your name space.

# Compute observed correlation: r_obs
r_obs = pearson_r(illiteracy, fertility)

# Initialize permutation replicates: perm_replicates
perm_replicates = np.empty(10000)

# Draw replicates
for i in range(10000):
    # Permute illiteracy measurments: illiteracy_permuted
    illiteracy_permuted = np.random.permutation(illiteracy)

    # Compute Pearson correlation
    perm_replicates[i] = pearson_r(illiteracy_permuted, fertility)

# Compute p-value: p
p = np.sum(perm_replicates >= r_obs) / len(perm_replicates)
print('p-val =', p)






###  BOOTSTRAP HYPOTHESIS TEST ON BEE SPERM COUNTS (comparison of means by bootstrap and p-value)

# Now, you will test the following hypothesis: On average, male bees treated with neonicotinoid 
# insecticide have the same number of active sperm per milliliter of semen than do untreated male 
# bees. You will use the difference of means as your test statistic.
# For your reference, the call signature for the draw_bs_reps() function you wrote in chapter 2 
# is draw_bs_reps(data, func, size=1).

# Compute the difference in mean sperm count: diff_means
diff_means = diff_of_means(np.mean(control), np.mean(treated))

# Compute mean of pooled data: mean_count
mean_count = np.mean(np.concatenate((control,treated)))


# Generate shifted data sets
control_shifted = control - np.mean(control) + mean_count
treated_shifted = treated - np.mean(treated) + mean_count

# Generate bootstrap replicates
bs_reps_control = draw_bs_reps(control_shifted,
                       np.mean, size=10000)
bs_reps_treated = draw_bs_reps(treated_shifted,
                       np.mean, size=10000)

# Get replicates of difference of means: bs_replicates
bs_replicates = bs_reps_control - bs_reps_treated

# Compute and print p-value: p
p = np.sum(bs_replicates >= np.mean(control) - np.mean(treated)) \
            / len(bs_replicates)
print('p-value =', p)

















############################ Chapter 5:




















###  EDA OF BEAK DEPTHS OF DARWIN'S FINCHES

# Create bee swarm plot
_ = sns.swarmplot(x='year',y='beak_depth',data=df)

# Label the axes
_ = plt.xlabel('year')
_ = plt.ylabel('beak depth (mm)')

# Show the plot
plt.show()



###  ECDFs OF BEAK DEPTHS

# Compute ECDFs
x_1975, y_1975 = ecdf(bd_1975)
x_2012, y_2012 = ecdf(bd_2012)

# Plot the ECDFs
_ = plt.plot(x_1975, y_1975, marker='.', linestyle='none')
_ = plt.plot(x_2012, y_2012, marker='.', linestyle='none')

# Set margins
plt.margins(0.02)

# Add axis labels and legend
_ = plt.xlabel('beak depth (mm)')
_ = plt.ylabel('ECDF')
_ = plt.legend(('1975', '2012'), loc='lower right')

# Show the plot
plt.show()




###  PARAMETER ESTIMATES OF BEAK DEPTHS (confidence intervals of different means)

# Compute the difference of the sample means: mean_diff
mean_diff = np.mean(bd_2012) - np.mean(bd_1975)

# Get bootstrap replicates of means
bs_replicates_1975 = draw_bs_reps(bd_1975, np.mean, size=10000)
bs_replicates_2012 = draw_bs_reps(bd_2012, np.mean, size=10000)

# Compute samples of difference of means: bs_diff_replicates
bs_diff_replicates = bs_replicates_2012 - bs_replicates_1975

# Compute 95% confidence interval: conf_int
conf_int = np.percentile(bs_diff_replicates,[2.5,97.5])

# Print the results
print('difference of means =', mean_diff, 'mm')
print('95% confidence interval =', conf_int, 'mm')





###  Hypothesis test: Are beaks deeper in 2012?

# Compute mean of combined data set: combined_mean
combined_mean = np.mean(np.concatenate((bd_1975, bd_2012)))

# Shift the samples
bd_1975_shifted = bd_1975 - np.mean(bd_1975) + combined_mean
bd_2012_shifted = bd_2012 - np.mean(bd_2012) + combined_mean

# Get bootstrap replicates of shifted data sets
bs_replicates_1975 = draw_bs_reps(bd_1975_shifted,np.mean, size=10000)
bs_replicates_2012 = draw_bs_reps(bd_2012_shifted,np.mean, size=10000)

# Compute replicates of difference of means: bs_diff_replicates
bs_diff_replicates = bs_replicates_2012 - bs_replicates_1975

# Compute the p-value
p = np.sum(bs_diff_replicates >= mean_diff) / len(bs_diff_replicates)

# Print p-value
print('p =', p)





###  EDA OF BEAK LENGTH AND DEPTH

# Make scatter plot of 1975 data
_ = plt.plot(bl_1975, bd_1975, marker='.',
             linestyle='none', color='blue', alpha=0.5)

# Make scatter plot of 2012 data
_ = plt.plot(bl_2012, bd_2012, marker='.',
             linestyle='none', color='red', alpha=0.5)

# Label axes and make legend
_ = plt.xlabel('beak length (mm)')
_ = plt.ylabel('beak depth (mm)')
_ = plt.legend(('1975', '2012'), loc='upper left')

# Show the plot
plt.show()





###  DISPLAYING THE LINEAR REGRESSION RESULTS  

# Make scatter plot of 1975 data
_ = plt.plot(bl_1975, bd_1975, marker='.',
             linestyle='none', color='blue', alpha=0.5)

# Make scatter plot of 2012 data
_ = plt.plot(bl_2012, bd_2012, marker='.',
             linestyle='none', color='red', alpha=0.5)

# Label axes and make legend
_ = plt.xlabel('beak length (mm)')
_ = plt.ylabel('beak depth (mm)')
_ = plt.legend(('1975', '2012'), loc='upper left')

# Generate x-values for bootstrap lines: x
x = np.array([10, 17])

# Plot the bootstrap lines
for i in range(100):
    plt.plot(x, bs_slope_reps_1975[i]*x + bs_intercept_reps_1975[i],
             linewidth=0.5, alpha=0.2, color='blue')
    plt.plot(x, bs_slope_reps_2012[i]*x + bs_intercept_reps_2012[i],
             linewidth=0.5, alpha=0.2, color='red')

# Draw the plot again
plt.show()





###  LINEAR REGRESSIONS

# Compute the linear regressions
slope_1975, intercept_1975 = np.polyfit(bl_1975, bd_1975, 1)
slope_2012, intercept_2012 = np.polyfit(bl_2012, bd_2012, 1)

# Perform pairs bootstrap for the linear regressions
bs_slope_reps_1975, bs_intercept_reps_1975 = \
        draw_bs_pairs_linreg(bl_1975, bd_1975, 1000)
bs_slope_reps_2012, bs_intercept_reps_2012 = \
        draw_bs_pairs_linreg(bl_2012, bd_2012, 1000)

# Compute confidence intervals of slopes
slope_conf_int_1975 = np.percentile(bs_slope_reps_1975,[2.5,97.5])
slope_conf_int_2012 = np.percentile(bs_slope_reps_2012,[2.5,97.5])
intercept_conf_int_1975 = np.percentile(bs_intercept_reps_1975,[2.5,97.5])
intercept_conf_int_2012 = np.percentile(bs_intercept_reps_2012,[2.5,97.5])


# Print the results
print('1975: slope =', slope_1975,
      'conf int =', slope_conf_int_1975)
print('1975: intercept =', intercept_1975,
      'conf int =', intercept_conf_int_1975)
print('2012: slope =', slope_2012,
      'conf int =', slope_conf_int_2012)
print('2012: intercept =', intercept_2012,
      'conf int =', intercept_conf_int_2012)




###  BEAK LENGTH TO DEPTH RATIO

# Compute length-to-depth ratios
ratio_1975 = bl_1975 / bd_1975
ratio_2012 = bl_2012 / bd_2012

# Compute means
mean_ratio_1975 = np.mean(ratio_1975)
mean_ratio_2012 = np.mean(ratio_2012)

# Generate bootstrap replicates of the means
bs_replicates_1975 = draw_bs_reps(ratio_1975, np.mean, size=10000)
bs_replicates_2012 = draw_bs_reps(ratio_2012, np.mean, size=10000)

# Compute the 99% confidence intervals
conf_int_1975 = np.percentile(bs_replicates_1975,[0.5,99.5])
conf_int_2012 = np.percentile(bs_replicates_2012,[0.5,99.5])

# Print the results
print('1975: mean ratio =', mean_ratio_1975,
      'conf int =', conf_int_1975)
print('2012: mean ratio =', mean_ratio_2012,
      'conf int =', conf_int_2012)




###  EDA OF HERITABILITY

# Make scatter plots
_ = plt.plot(bd_parent_fortis, bd_offspring_fortis,
             marker='.', linestyle='none', color='blue', alpha=0.5)
_ = plt.plot(bd_parent_scandens, bd_offspring_scandens,
             marker='.', linestyle='none', color='red', alpha=0.5)

# Set margins
plt.margins(0.02)

# Label axes
_ = plt.xlabel('parental beak depth (mm)')
_ = plt.ylabel('offspring beak depth (mm)')

# Add legend
_ = plt.legend(('G. fortis', 'G. scandens'), loc='lower right')

# Show plot
plt.show()






### Correlation of offspring and parental data

# In an effort to quantify the correlation between offspring and parent 
# beak depths, we would like to compute statistics, such as the Pearson 
# correlation coefficient, between parents and offspring. To get confidence 
# intervals on this, we need to do a pairs bootstrap.

def draw_bs_pairs(x, y, func,size=1):
    """Perform pairs bootstrap for linear regression."""

    # Set up array of indices to sample from: inds
    inds = np.arange(len(x))

    # Initialize replicates
    bs_replicates = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_inds = np.random.choice(inds, len(inds))
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        bs_replicates[i] = func(bs_x, bs_y)

    return bs_replicates






###  Pearson correlation of offspring and parental data

def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    # Compute correlation matrix: corr_mat
    corr_mat = np.corrcoef(x,y)

    # Return entry [0,1]
    return corr_mat[0,1]



# Compute the Pearson correlation coefficients
r_scandens = pearson_r(bd_parent_scandens, bd_offspring_scandens)
r_fortis = pearson_r(bd_parent_fortis, bd_offspring_fortis)

# Acquire 1000 bootstrap replicates of Pearson r
bs_replicates_scandens = draw_bs_pairs(bd_parent_scandens, bd_offspring_scandens, pearson_r, 1000)

bs_replicates_fortis = draw_bs_pairs(bd_parent_fortis, bd_offspring_fortis, pearson_r, 1000)


# Compute 95% confidence intervals
conf_int_scandens = np.percentile(bs_replicates_scandens,[2.5,97.5])
conf_int_fortis = np.percentile(bs_replicates_fortis,[2.5,97.5])

# Print results
print('G. scandens:', r_scandens, conf_int_scandens)
print('G. fortis:', r_fortis, conf_int_fortis)





###  Measuring heritability

# Remember that the Pearson correlation coefficient is the ratio of the covariance 
# to the geometric mean of the variances of the two data sets. This is a measure of 
# the correlation between parents and offspring, but might not be the best estimate 
# of heritability. If we stop and think, it makes more sense to define heritability 
# as the ratio of the covariance between parent and offspring to the variance of 
# the parents alone. In this exercise, you will estimate the heritability and 
# perform a pairs bootstrap calculation to get the 95% confidence interval.

def heritability(parents, offspring):
    """Compute the heritability from parent and offspring samples."""
    covariance_matrix = np.cov(parents, offspring)
    return covariance_matrix[0,1]/ covariance_matrix[0,0]

# Compute the heritability
heritability_scandens = heritability(bd_parent_scandens,bd_offspring_scandens)
heritability_fortis = heritability(bd_parent_fortis, bd_offspring_fortis)

# Acquire 1000 bootstrap replicates of heritability
replicates_scandens = draw_bs_pairs(
        bd_parent_scandens, bd_offspring_scandens, heritability, size=1000)
        
replicates_fortis = draw_bs_pairs(
        bd_parent_fortis, bd_offspring_fortis, heritability, size=1000)

# Compute 95% confidence intervals
conf_int_scandens = np.percentile(replicates_scandens,[2.5,97.5])
conf_int_fortis = np.percentile(replicates_fortis,[2.5,97.5])

# Print results
print('G. scandens:', heritability_scandens, conf_int_scandens)
print('G. fortis:', heritability_fortis, conf_int_fortis)





###  Is beak depth heritable at all in *G. scandens*?

# The heritability of beak depth in G. scandens seems low. It could be that 
# this observed heritability was just achieved by chance and beak depth is 
# actually not really heritable in the species. You will test that hypothesis 
# here. To do this, you will do a pairs permutation test.

# Initialize array of replicates: perm_replicates
perm_replicates = np.empty(10000)

# Draw replicates
for i in range(10000):
    # Permute parent beak depths
    bd_parent_permuted = np.random.permutation(bd_parent_scandens)
    perm_replicates[i] = heritability(bd_parent_permuted, bd_offspring_scandens)

# Compute p-value: p
p = np.sum(perm_replicates >= heritability_scandens) / len(perm_replicates)

# Print the p-value
print('p-val =', p)



