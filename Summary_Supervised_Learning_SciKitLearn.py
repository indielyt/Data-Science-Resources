
### Supervised Learning with SciKit Learn

# 1. Classification

		# Exploratory Data Analysis on continuous data (iris dataset)
		# Exploratory Data Analysis on Binary data
		# k-Nearest Neighbors: Fit
		# k-Nearest Neighbors: Predict
		# The digits recognition dataset
		# Train/Test Split + Fit/Predict/Accuracy using K-nearest neighbors
		# Overfitting and underfitting (determining how many neighbors(k))

# 2. Regression

		# Importing data for supervised learning
		# Exploring the Gapminder data
		# Fit & predict for linear regression (see later for regularization)
		# Train/test split for regression
		# 5-fold cross-validation
		# K-Fold CV comparison
		# Regularization I: Lasso
		# Regularization II: Ridge

# 3. Fine Tuning Models

		# Metrics for classification (confusion matrix, precision, recall, F1 scores)
		# Building a logistic regression model
		# Plotting an ROC curve (Receiver Operating Characteristic Curve)
		# AUC computation (Area Under the Curve)
		# Hyperparameter tuning with GridSearchCV
		# Hyperparameter tuning with RandomizedSearchCV
		# Hold-out set in practice I: Classification
		# Hold-out set in practice II: Regression	

# 4. Preprocessing and Pipelines

		# Exploring categorical features (boxplots)
		# Creating dummy variables
		# Regression with categorical features
		# Dropping missing data
		# Imputing missing data in a ML Pipeline I
		# Imputing missing data in a ML Pipeline II
		# Centering and scaling your data
		# Centering and scaling in a pipeline
		# Bringing it all together I: Pipeline for classification
		# Bringing it all together II: Pipeline for regression



###########################

from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set plot style
plt.style.use('ggplot')



### Exploratory Data Analysis on continuous data

# Import iris dataset and investigate basics
iris = datasets.load_iris()
type (iris)
print (iris.keys)
type (iris.data), type(iris.target)
iris.data.shape
iris.target_names

# Exploratory Data Analysis (EDA)
X = iris.data #Set feature variable from numpy array
y = iris.target #Set target variable from numpy array
df = pd.DataFrame(X, columns=iris.feature_names) #transform to pandas dataframe
print (df.head()) #Examine first five dataframe features

# Visual EDA with scatter matrix.  
#'c' refers to color by target variable, 's' refers to size
_ = pd.scatter_matrix(df, c=y, figsize=[8,8], s=150, marker='D')
plt.show()





### Exploratory Data Analysis on Binary data

plt.figure()
sns.countplot(x='education', hue='party', data=df, palette='RdBu')
plt.xticks([0,1], ['No', 'Yes'])
plt.show()




### k-Nearest Neighbors: Fit

# Import KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier

# Create arrays for the features and the response variable
y = df['party'].values # define the target (labels) variable
X = df.drop('party', axis=1).values # drop the target (labels) variable

# Create a k-NN classifier with 6 neighbors
knn = KNeighborsClassifier(n_neighbors=6) # initiate a k-nearest neighbors classifier 
# Scikit Learn is built primarily using python classes

# Fit the classifier to the data
knn.fit(X,y)





### k-Nearest Neighbors: Predict

# Import KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier 

# Create arrays for the features and the response variable
y = df['party'].values
X = df.drop('party',axis=1).values

# Create a k-NN classifier with 6 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the data
knn.fit(X,y)

# Predict the labels for the training data X
y_pred = knn.predict(X)

# Predict and print the label for the new data point X_new
new_prediction = knn.predict(X_new)
print("Prediction: {}".format(new_prediction))





### The digits recognition dataset

# Import necessary modules
from sklearn import datasets
import matplotlib.pyplot as plt

# Load the digits dataset: digits
digits = datasets.load_digits()

# Print the keys and DESCR of the dataset
print(digits.keys())
print(digits.DESCR)

# Print the shape of the images and data keys
print(digits.images.shape)
print(digits.data.shape)

# Display digit 1010
plt.imshow(digits.images[1010], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()





### Train/Test Split + Fit/Predict/Accuracy using K-nearest neighbors

# Import necessary modules
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split 

# Create feature and target arrays
X = digits.data
y = digits.target

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)

# Create a k-NN classifier with 7 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=7)

# Fit the classifier to the training data
knn.fit(X_train,y_train)

# Print the accuracy
print(knn.score(X_test, y_test))





### Overfitting and underfitting (determining how many neighbors(k))

# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the training data
    knn.fit(X_train,y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()







##########################








from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split 

# Set plot style
plt.style.use('ggplot')




### Importing data for supervised learning

# Import numpy and pandas
import numpy as np
import pandas as pd

# Read the CSV file into a DataFrame: df
df = pd.read_csv('gapminder.csv')

# Create arrays for features and target variable
y = df['life'].values
X = df['fertility'].values

# Print the dimensions of X and y before reshaping
print("Dimensions of y before reshaping: {}".format(y.shape))
print("Dimensions of X before reshaping: {}".format(X.shape))

# Reshape X and y
y = y.reshape(-1,1)
X = X.reshape(-1,1)

# Print the dimensions of X and y after reshaping
print("Dimensions of y after reshaping: {}".format(y.shape))
print("Dimensions of X after reshaping: {}".format(X.shape))




### Exploring the Gapminder data

# Determine columns, number of rows, and data types
df.info()

# Count, mean, std dev, min, 25/50/75 percentiles, and max for each column
df.describe()

# see first 5 rows
df.head()

# explore correlation of variables by pair-wise comparison, then create heatmap
df.corr()
sns.heatmap(df.corr(), square=True, cmap='RdYlGn')





### Fit & predict for linear regression (see later for regularization)

# Import LinearRegression
from sklearn.linear_model import LinearRegression 

# Create the regressor: reg
reg = LinearRegression()

# Create the prediction space
prediction_space = np.linspace(min(X_fertility), max(X_fertility)).reshape(-1,1)

# Fit the model to the data
reg.fit(X_fertility,y)

# Compute predictions over the prediction space: y_pred
y_pred = reg.predict(prediction_space)

# Print R^2 
print(reg.score(X_fertility, y))

# Plot regression line
plt.plot(prediction_space, y_pred, color='black', linewidth=3)
plt.show()





### Train/test split for regression

# Import necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=42)

# Create the regressor: reg_all
reg_all = LinearRegression()

# Fit the regressor to the training data
reg_all.fit(X_train,y_train)

# Predict on the test data: y_pred
y_pred = reg_all.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2: {}".format(reg_all.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error: {}".format(rmse))





### 5-fold cross-validation

# Import the necessary modules
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

# Create a linear regression object: reg
reg = LinearRegression()

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg,X,y,cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))





### K-Fold CV comparison

# Import necessary modules
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import cross_val_score 

# Create a linear regression object: reg
reg = LinearRegression()

# Perform 3-fold CV
cvscores_3 = cross_val_score(reg,X,y,cv=3)
print(np.mean(cvscores_3))

# Perform 10-fold CV
cvscores_10 = cross_val_score(reg,X,y,cv=10)
print(np.mean(cvscores_10))





### Regularization I: Lasso

# Import Lasso
from sklearn.linear_model import Lasso

# Instantiate a lasso regressor: lasso
lasso = Lasso(alpha=0.4, normalize=True)

# Fit the regressor to the data
lasso.fit(X,y)

# Compute and print the coefficients
lasso_coef = lasso.coef_
print(lasso_coef)

# Plot the coefficients
plt.plot(range(len(df_columns)), lasso_coef)
plt.xticks(range(len(df_columns)), df_columns.values, rotation=60)
plt.margins(0.02)
plt.show()





### Regularization II: Ridge

# define function for visualizing scores and std deviations
def display_plot(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)
    
    std_error = cv_scores_std / np.sqrt(10)
    
    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.show()

# Import necessary modules
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Setup the array of alphas and lists to store scores
alpha_space = np.logspace(-4, 0, 50)
ridge_scores = []
ridge_scores_std = []

# Create a ridge regressor: ridge
ridge = Ridge(normalize=True)

# Compute scores over range of alphas
for alpha in alpha_space:

    # Specify the alpha value to use: ridge.alpha
    ridge.alpha = alpha
    
    # Perform 10-fold CV: ridge_cv_scores
    ridge_cv_scores = cross_val_score(ridge, X, y, cv=10)
    
    # Append the mean of ridge_cv_scores to ridge_scores
    ridge_scores.append(np.mean(ridge_cv_scores))
    
    # Append the std of ridge_cv_scores to ridge_scores_std
    ridge_scores_std.append(np.std(ridge_cv_scores))

# Display the plot
display_plot(ridge_scores, ridge_scores_std)





####################################









from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set plot style
plt.style.use('ggplot')





### Metrics for classification (confusion matrix, precision, recall, F1 scores)

# Import necessary modules
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Create training and test set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4, random_state=42)

# Instantiate a k-NN classifier: knn
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Predict the labels of the test data: y_pred
y_pred = knn.predict(X_test)

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))





### Building a logistic regression model

# Import the necessary modules
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import confusion_matrix, classification_report

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)

# Create the classifier: logreg
logreg = LogisticRegression()

# Fit the classifier to the training data
logreg.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = logreg.predict(X_test)

# Compute and print the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))





### Plotting an ROC curve (Receiver Operating Characteristic Curve)

# One way to understand this curve is to think of it as the performance of
# predicting judging true positives vs. the likelihood of predicting false
# positives (x and y axes on the graph).  A perfect model would have just one
# point at location (0,1), or said a different way, 0% chance of predicting
# false positives, 100% chance of predicting true positives.  The AUC curve in the 
# next exercise helps quantify the ROC curve insights about model performance

# Import necessary modules
from sklearn.metrics import roc_curve

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr (false positive rate), tpr (true positive rate), thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()




### AUC computation (Area Under the Curve)

# Say you have a binary classifier that in fact is just randomly making guesses. 
# It would be correct approximately 50% of the time, and the resulting ROC curve 
# would be a diagonal line in which the True Positive Rate and False Positive Rate 
# are always equal. The Area under this ROC curve would be 0.5. This is one way in 
# which the AUC, which Hugo discussed in the video, is an informative metric to 
# evaluate a model. If the AUC is greater than 0.5, the model is better than random 
# guessing. Always a good sign!

# Import necessary modules
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Compute and print AUC score
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))

# Compute cross-validated AUC scores: cv_auc
cv_auc = cross_val_score(logreg,X,y,cv=5,scoring='roc_auc')

# Print list of AUC scores
print("AUC scores computed using 5-fold cross-validation: {}".format(cv_auc))





### Hyperparameter tuning with GridSearchCV

# Hugo demonstrated how to use to tune the n_neighbors parameter of the KNeighborsClassifier() 
# using GridSearchCV on the voting dataset. You will now practice this yourself, but by using 
# logistic regression on the diabetes dataset instead!  Like the alpha parameter of lasso and 
# ridge regularization that you saw earlier, logistic regression also has a regularization 
# parameter: CC. CC controls the inverse of the regularization strength, and this is what you 
# will tune in this exercise. A large CC can lead to an overfit model, while a small CC can 
# lead to an underfit model.  The hyperparameter space for CC has been setup for you. 
# Your job is to use GridSearchCV and logistic regression to find the optimal CC in this 
# hyperparameter space. The feature array is available as X and target variable array is 
# available as y.

# This method uses the grid search module which compares the range of possible parameter values 
# to the cross validation scores.  The best is returned as best_params_ and best_score_.  This
# example is for one variable, but the method applies to multi-parameter models as well.

# Import necessary modules
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import GridSearchCV 

# Setup the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space}

# Instantiate a logistic regression classifier: logreg
logreg = LogisticRegression()

# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)

# Fit it to the data
logreg_cv.fit(X,y)

# Print the tuned parameters and score
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_)) 
print("Best score is {}".format(logreg_cv.best_score_))






# Hyperparameter tuning with RandomizedSearchCV

# GridSearchCV can be computationally expensive, especially if you are searching over a 
# large hyperparameter space and dealing with multiple hyperparameters. A solution to 
# this is to use RandomizedSearchCV, in which not all hyperparameter values are tried 
# out. Instead, a fixed number of hyperparameter settings is sampled from specified 
# probability distributions. You'll practice using RandomizedSearchCV in this exercise 
# and see how this works. Here, you'll also be introduced to a new model: the Decision 
# Tree. Don't worry about the specifics of how this model works. Just like k-NN, linear 
# regression, and logistic regression, decision trees in scikit-learn have .fit() and 
# .predict() methods that you can use in exactly the same way as before. Decision trees 
# have many parameters that can be tuned, such as max_features, max_depth, and 
# min_samples_leaf: This makes it an ideal use case for RandomizedSearchCV.  Note that 
# RandomizedSearchCV will never outperform GridSearchCV. Instead, it is valuable because 
# it saves on computation time.

# Import necessary modules
from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV

# Setup the parameters and distributions to sample from: param_dist
param_dist = {"max_depth": [3, None],
              "max_features": randint(1, 9),
              "min_samples_leaf": randint(1, 9),
              "criterion": ["gini", "entropy"]}

# Instantiate a Decision Tree classifier: tree
tree = DecisionTreeClassifier()

# Instantiate the RandomizedSearchCV object: tree_cv
tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)

# Fit it to the data
tree_cv.fit(X,y)

# Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))





# Hold-out set in practice I: Classification

# You will now practice evaluating a model with tuned hyperparameters on a 
# hold-out set. The feature array and target variable array from the diabetes 
# dataset have been pre-loaded as X and y. In addition to C, logistic regression 
# has a 'penalty' hyperparameter which specifies whether to use 'l1' or 'l2' 
# regularization. Your job in this exercise is to create a hold-out set, tune 
# the 'C' and 'penalty' hyperparameters of a logistic regression classifier 
# using GridSearchCV on the training set, and then evaluate its performance 
# against the hold-out set.

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Create the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space, 'penalty': ['l1', 'l2']}

# Instantiate the logistic regression classifier: logreg
logreg = LogisticRegression()

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4, random_state=42)

# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)

# Fit it to the training data
logreg_cv.fit(X_train, y_train)

# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter: {}".format(logreg_cv.best_params_))
print("Tuned Logistic Regression Accuracy: {}".format(logreg_cv.best_score_))





# Hold-out set in practice II: Regression

#Remember lasso and ridge regression from the previous chapter? Lasso used the L1 
# penalty to regularize, while ridge used the L2 penalty. There is another type of 
# regularized regression known as the elastic net. In elastic net regularization, the 
# penalty term is a linear combination of the L1L1 and L2L2 penalties:

# a∗L1+b∗L2
# a∗L1+b∗L2

# In scikit-learn, this term is represented by the 'l1_ratio' parameter: An 'l1_ratio' 
# of 1 corresponds to an L1 penalty, and anything lower is a combination of L1 and L2.
# In this exercise, you will GridSearchCV to tune the 'l1_ratio' of an elastic net model trained 
# on the Gapminder data. As in the previous exercise, use a hold-out set to evaluate 
# your model's performance. 

# Import necessary modules
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4, random_state=42)

# Create the hyperparameter grid
l1_space = np.linspace(0, 1, 30)
param_grid = {'l1_ratio':l1_space}

# Instantiate the ElasticNet regressor: elastic_net
elastic_net = ElasticNet()

# Setup the GridSearchCV object: gm_cv
gm_cv = GridSearchCV(elastic_net, param_grid, cv=5)

# Fit it to the training data
gm_cv.fit(X_train, y_train)

# Predict on the test set and compute metrics
y_pred = gm_cv.predict(X_test)
r2 = gm_cv.score(X_test, y_test)
mse = mean_squared_error(y_test, y_pred)
print("Tuned ElasticNet l1 ratio: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))
print("Tuned ElasticNet MSE: {}".format(mse))








############################











from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set plot style
plt.style.use('ggplot')





### Exploring categorical features (boxplots)

# Import pandas
import pandas as pd

# Read 'gapminder.csv' into a DataFrame: df
df = pd.read_csv('gapminder.csv')

# Create a boxplot of life expectancy per region
df.boxplot('life', 'Region', rot=60)

# Show the plot
plt.show()





### Creating dummy variables

# This method adds columns of categorical data columns with row values
# of 1 or 0 to indicate what category the rest of the row belongs to.  In 
# this example, 'regions' were converted to 4 columns, each column being a 
# unique region.  The rows in these columns have 1s or 0s that indicate if 
# which region the rest of that rows data belongs to

# Create dummy variables: df_region
df_region = pd.get_dummies(df)

# Print the columns of df_region
print(df_region.columns)

# Create dummy variables with drop_first=True: df_region
df_region = pd.get_dummies(df,drop_first=True)

# Print the new columns of df_region
print(df_region.columns)





### Regression with categorical features

# Import necessary modules
from sklearn.linear_model import Ridge 
from sklearn.model_selection import cross_val_score 

# Instantiate a ridge regressor: ridge
ridge = Ridge(alpha=0.5, normalize=True)

# Perform 5-fold cross-validation: ridge_cv
ridge_cv = cross_val_score(ridge,X,y,cv=5)

# Print the cross-validated scores
print(ridge_cv)





### Dropping missing data

# Note this drops rows with any missing data, a lot can be lost in this way

# Convert '?' to NaN
df[df == '?'] = np.nan

# Print the number of NaNs
print(df.isnull().sum())

# Print shape of original DataFrame
print("Shape of Original DataFrame: {}".format(df.shape))

# Drop missing values and print shape of new DataFrame
df = df.dropna()

# Print shape of new DataFrame
print("Shape of DataFrame After Dropping All Rows with Missing Values: {}".format(df.shape))





### Imputing missing data in a ML Pipeline I

# Depending on the dataset, assigning a value to missing data may be superior to throwing
# out the entire observation, imputataion does this in a number of user-controlled ways

# Import the Imputer module
from sklearn.preprocessing import Imputer
from sklearn.svm import SVC

# Setup the Imputation transformer: imp
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)

# Instantiate the SVC classifier: clf
clf = SVC()

# Setup the pipeline with the required steps: steps
steps = [('imputation', imp),
        ('SVM', clf)]





### Imputing missing data in a ML Pipeline II

# Pipelines help automate the steps of a supervised learning workflow.
# Here we impute the missing values with the 'most_frequent' method, and
# classify the multi-dimensional data into binary republican or democrat groups
# (see the steps tuple). The pipeline object can be used for fit and predict
# A classification report expresses the score of the model.

# Import necessary modules
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Setup the pipeline steps: steps
steps = [('imputation', Imputer(missing_values='NaN', strategy='most_frequent', axis=0)),
        ('SVM', SVC())]

# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

# Fit the pipeline to the train set
pipeline.fit(X_train,y_train)

# Predict the labels of the test set
y_pred = pipeline.predict(X_test)

# Compute metrics
print(classification_report(y_test, y_pred))





### Centering and scaling your data

# Import scale
from sklearn.preprocessing import scale

# Scale the features: X_scaled
X_scaled = scale(X)

# Print the mean and standard deviation of the unscaled features
print("Mean of Unscaled Features: {}".format(np.mean(X))) 
print("Standard Deviation of Unscaled Features: {}".format(np.std(X)))

# Print the mean and standard deviation of the scaled features
print("Mean of Scaled Features: {}".format(np.mean(X_scaled))) 
print("Standard Deviation of Scaled Features: {}".format(np.std(X_scaled)))




### Centering and scaling in a pipeline

# Import the necessary modules
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Setup the pipeline steps: steps
steps = [('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())]
        
# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

# Fit the pipeline to the training set: knn_scaled
knn_scaled = pipeline.fit(X_train, y_train)

# Instantiate and fit a k-NN classifier to the unscaled data
knn_unscaled = KNeighborsClassifier().fit(X_train, y_train)

# Compute and print metrics
print('Accuracy with Scaling: {}'.format(knn_scaled.score(X_test,y_test)))
print('Accuracy without Scaling: {}'.format(knn_unscaled.score(X_test,y_test)))






### Bringing it all together I: Pipeline for classification

# Setup the pipeline
steps = [('scaler', StandardScaler()),
         ('SVM', SVC())]

pipeline = Pipeline(steps)

# Specify the hyperparameter space
parameters = {'SVM__C':[1, 10, 100],
              'SVM__gamma':[0.1, 0.01]}

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=21)

# Instantiate the GridSearchCV object: cv
cv = GridSearchCV(pipeline, parameters)

# Fit to the training set
cv.fit(X_train,y_train)

# Predict the labels of the test set: y_pred
y_pred = cv.predict(X_test)

# Compute and print metrics
print("Accuracy: {}".format(cv.score(X_test, y_test)))
print(classification_report(y_test, y_pred))
print("Tuned Model Parameters: {}".format(cv.best_params_))






### Bringing it all together II: Pipeline for regression

# Setup the pipeline steps: steps
steps = [('imputation', Imputer(missing_values='NaN', strategy='mean', axis=0)),
         ('scaler', StandardScaler()),
         ('elasticnet', ElasticNet())]

# Create the pipeline: pipeline 
pipeline = Pipeline(steps)

# Specify the hyperparameter space
parameters = {'elasticnet__l1_ratio':np.linspace(0,1,30)}

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, 
random_state=42)

# Create the GridSearchCV object: gm_cv
gm_cv = GridSearchCV(pipeline, parameters)

# Fit to the training set
gm_cv.fit(X_train, y_train)

# Compute and print the metrics
r2 = gm_cv.score(X_test, y_test)
print("Tuned ElasticNet Alpha: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))