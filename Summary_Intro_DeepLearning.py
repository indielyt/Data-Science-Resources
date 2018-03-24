### DATACAMP INTRODUCTION DEEP LEARNING

import numpy as np

# Chapter 1: Basics of deep learning and neural networks

	# Coding the forward propagation algorithm
	# The Rectified Linear Activation Function
	# Applying the network to many observations/rows of data
	# Multi-layer neural networks

# Chapter 2: Optimizing a neural network with backward propagation

	# Coding how weight changes affect accuracy
	# Scaling up to multiple data points
	# Calculating Slopes with a single data point
    # Improving model weights
    # Making multiple updates to weights

# Chapter 3: Building deep learning models with Keras

    # Specifying a model
    # Compiling the model
    # Fitting the model
    # Last steps in classification models
    # Making predictions 

# Chapter 4: Fine Tuning Keras Models

    # Changing optimization parameters
    # Evaluating model accuracy on validation dataset
    # Early stopping: Optimizing the optimization
    # Experimenting with wider networks
    # Adding layers to a network
    # Building your own digit recognition model














# CHAPTER 1 ###################################################################################### 











### Coding the forward propagation algorithm

# Array of input data, number of accounts (input_data[0]) and number of children (input_data[1])
input_data = np.array([3, 5])

# Dictionary of weights for the two hidden nodes and output, see svg
weights = {'node_0': array([2, 4]), 'node_1': array([ 4, -5]), 'output': array([2, 7])}

# Calculate node 0 value: node_0_value
# hidden node values are the input data multiplied by the weights of the input data, then summed
node_0_value = (input_data * weights['node_0']).sum()

# Calculate node 1 value: node_1_value
node_1_value = (input_data * weights['node_1']).sum()

# Put node values into array: hidden_layer_outputs
hidden_layer_outputs = np.array([node_0_value, node_1_value])

# Calculate output: output
output = (hidden_layer_outputs * weights['output']).sum()

# Print output
print(output)

#Output:
-39






### The Rectified Linear Activation Function

# Defining a function to alter the output of the node improves model 
# performance by allowing non-linear relationships to be passed through
# the hidden nodes. Here we improve on the previous example by taking
# negative node values and passing zero values forward instead. 

# Array of input data, number of accounts (input_data[0]) and number of children (input_data[1])
input_data = np.array([3, 5])

# Dictionary of weights for the two hidden nodes and output, see svg
weights = {'node_0': array([2, 4]), 'node_1': array([ 4, -5]), 'output': array([2, 7])}

def relu(input):
    '''Define your relu activation function here'''
    # Calculate the value for the output of the relu function: output
    output = max(input, 0)
    
    # Return the value just calculated
    return(output)

# Calculate node 0 value: node_0_output
node_0_input = (input_data * weights['node_0']).sum()
node_0_output = relu(node_0_input)

# Calculate node 1 value: node_1_output
node_1_input = (input_data * weights['node_1']).sum()
node_1_output = relu(node_1_input)

# Put node values into array: hidden_layer_outputs
hidden_layer_outputs = np.array([node_0_output, node_1_output])

# Calculate model output (do not apply relu)
model_output = (hidden_layer_outputs * weights['output']).sum()

# Print model output
print(model_output)







### Applying the network to many observations/rows of data

# We'll reuse the same rectified linear activation function as before, and 
# make a prediction for each row (observation) of an input data table by calling that
# function in a for loop. The output is a prediction of number of transactions
# for the 4 rows (different bank customers).

# Input data with multiple observations
input_data = [array([3, 5]), array([ 1, -1]), array([0, 0]), array([8, 4])]

# Weights
weights = {'node_0': array([2, 4]), 'node_1': array([ 4, -5]), 'output': array([2, 7])}

# Define predict_with_network()
def predict_with_network(input_data_row, weights):

    # Calculate node 0 value
    node_0_input = (input_data_row*weights['node_0']).sum()
    node_0_output = relu(node_0_input)

    # Calculate node 1 value
    node_1_input = (input_data_row*weights['node_1']).sum()
    node_1_output = relu(node_1_input)

    # Put node values into array: hidden_layer_outputs
    hidden_layer_outputs = np.array([node_0_output, node_1_output])
    
    # Calculate model output
    input_to_final_layer = (hidden_layer_outputs*weights['output']).sum() 
    model_output = relu(input_to_final_layer)
    
    # Return model output
    return(model_output)


# Create empty list to store prediction results
results = []
for input_data_row in input_data:
    # Append prediction to results
    results.append(predict_with_network(input_data_row,weights))

# Print results
print(results)

# Output:
[52, 63, 0, 148]





### Multi-layer neural networks

# this code will perform forward propogation for a neural network with two hidden layers (see svg)

# Input data
input_data = array([3, 5])

# First hidden layer weights (describe weights associated with input data to this layer)
weights['node_0_0'] = array([2, 4])
weights['node_0_1'] = array([ 4, -5])

# Second hidden layer weights (describe weights associated with first hidden layer nodes to this layer)
weights['node_1_0'] = array([-1,  2])
weights['node_1_1'] = array([1, 2])

# Weights for the output layer
weights['output'] = array([2, 7])

def predict_with_network(input_data):
    # Calculate node 0 in the first hidden layer
    node_0_0_input = (input_data * weights['node_0_0']).sum()
    node_0_0_output = relu(node_0_0_input)

    # Calculate node 1 in the first hidden layer
    node_0_1_input = (input_data * weights['node_0_1']).sum()
    node_0_1_output = (relu(node_0_1_input))

    # Put node values into array: hidden_0_outputs
    hidden_0_outputs = np.array([node_0_0_output, node_0_1_output])
    
    # Calculate node 0 in the second hidden layer
    node_1_0_input = (hidden_0_outputs * weights['node_1_0']).sum()
    node_1_0_output = relu(node_1_0_input)

    # Calculate node 1 in the second hidden layer
    node_1_1_input = (hidden_0_outputs * weights['node_1_1']).sum()
    node_1_1_output = relu(node_1_1_input)

    # Put node values into array: hidden_1_outputs
    hidden_1_outputs = np.array([node_1_0_output, node_1_1_output])

    # Calculate model output: model_output
    model_output = (hidden_1_outputs * weights['output']).sum() 
    
    # Return model_output
    return(model_output)

output = predict_with_network(input_data)
print(output)

# Output:
182














# CHAPTER 2 ###################################################################################### 













### Coding how weight changes affect accuracy

# Here we manually change one weight to get a perfect prediction
# for a single observation (features and target value)

# The data point you will make a prediction for
input_data = np.array([0, 3])

# Sample weights
weights_0 = {'node_0': [2, 1],
             'node_1': [1, 2],
             'output': [1, 1]
            }

# The actual target value, used to calculate the error
target_actual = 3

# Make prediction using original weights
model_output_0 = predict_with_network(input_data, weights_0)

# Calculate error: error_0
error_0 = model_output_0 - target_actual

# Create weights that cause the network to make perfect prediction (3): weights_1
weights_1 = {'node_0': [2, 1],
             'node_1': [1, 0],
             'output': [1, 1]
            }

# Make prediction using new weights: model_output_1
model_output_1 = predict_with_network(input_data, weights_1)

# Calculate error: error_1
error_1 = model_output_1 - target_actual

# Print error_0 and error_1
print(error_0)
print(error_1)

# Output:
6
0





### Scaling up to multiple data points

# Calculating model accuracy for more than one data point, using the version
# of the previouslydefined predict_with_network() function.  Here the function
# takes two arguments - input data and weights.

# Input data, 4 unique observations
input_data = [array([0, 3]), array([1, 2]), array([-1, -2]), array([4, 0])]

# Weights
weights_0 = {'node_0': array([2, 1]), 'node_1': array([1, 2]), 'output': array([1, 1])}
weights_1 = {'node_0': array([2, 1]), 'node_1': array([ 1 , 1.5]), 'output': array([ 1 ,  1.5])}

# Known target values
target_actuals = [1, 3, 5, 7]

from sklearn.metrics import mean_squared_error

# Create model_output_0 
model_output_0 = []
# Create model_output_1
model_output_1 = []

# Loop over input_data
for row in input_data:
    # Append prediction to model_output_0
    model_output_0.append(predict_with_network(row, weights_0))
    
    # Append prediction to model_output_1
    model_output_1.append(predict_with_network(row, weights_1))

# Calculate the mean squared error for model_output_0: mse_0
mse_0 = mean_squared_error(target_actuals, model_output_0)

# Calculate the mean squared error for model_output_1: mse_1
mse_1 = mean_squared_error(target_actuals, model_output_1)

# Print mse_0 and mse_1
print("Mean squared error with weights_0: %f" %mse_0)
print("Mean squared error with weights_1: %f" %mse_1)

# Output:
Mean squared error with weights_0: 294.000000
Mean squared error with weights_1: 395.062500





### Calculating Slopes with a single data point

# Using a single data point with associated values, we'll 
# calculate the slope for the weight. The slope for the weight
# is the product of three items: (1) the slope of the error function
# with respect to the node we feed into, in this case the MSE - mean squared error.
# For a single data point where n=1, MSE is ((prediction-target)^2)/n.  The slope is 
# the first derivative; f'(x) = 2 * (prediction-target). (2) The value of the
# node that feeds into our weight, and (3) the slope of the activation 
# function with respect the the value we feed into.  In this case the
# activation function is the identity function - or more plainly 
# said just the value of 1. 

weights = ([0, 2, 1])
input_data = array([1, 2, 3])
target = 0

# Calculate the predictions: preds
preds = (weights * input_data).sum()

# Calculate the error: error
error = preds - target
print ('error=', error)

# Calculate the slope: slope
slope = 2 * input_data * error

# Print the slope
print(slope)

# Output:
[14 28 42]





### Improving model weights

# Use the slopes and a learning rate to recalculate weights
# and improve the model's error

# Set the learning rate: learning_rate
learning_rate = 0.01

# Calculate the predictions: preds
preds = (weights * input_data).sum()

# Calculate the error: error
error = preds - target

# Calculate the slope: slope
slope = 2 * input_data * error

# Update the weights: weights_updated
weights_updated = weights - learning_rate*(slope)

# Get updated predictions: preds_updated
preds_updated = (weights_updated * input_data).sum()

# Calculate updated error: error_updated
error_updated = preds_updated - target

# Print the original error
print(error)

# Print the updated error
print(error_updated)





### Making multiple updates to weights

# You're now going to make multiple updates so you can dramatically improve 
# your model weights, and see how the predictions improve with each update.

# Note: getting the function code out of datacamp website with the following code:
# import inspect
# lines = inspect.getsourcelines(foo)
# print("".join(lines[0]))

def get_error(input_data, target, weights):
    preds = (weights * input_data).sum()
    error = preds - target
    return(error)

def get_mse(input_data, target, weights):
    errors = get_error(input_data, target, weights)
    mse = np.mean(errors**2)
    return(mse)

def get_slope(input_data, target, weights):
    error = get_error(input_data, target, weights)
    slope = 2 * input_data * error
    return(slope)


n_updates = 20
mse_hist = []

# Iterate over the number of updates
for i in range(n_updates):
    # Calculate the slope: slope
    slope = get_slope(input_data, target, weights)
    
    # Update the weights: weights
    weights = weights - 0.01 * slope
    
    # Calculate mse with new weights: mse
    mse = get_mse(input_data, target, weights)
    
    # Append the mse to mse_hist
    mse_hist.append(mse)

# Plot the mse history
plt.plot(mse_hist)
plt.xlabel('Iterations')
plt.ylabel('Mean Squared Error')
plt.show()


















# CHAPTER 3 ######################################################################################

















### Specifying a model

# Building a model using Keras involves 4 steps: (1) Specifying network architecture. 
# (2) Compiling the model. (3) Fit the model. (4) Predict values

#  Sequential neural networks have each layer connecting only
# the the next layer, rather than to layers further along the model structure.  Dense implies
# all nodes in the layer connect to all nodes in the next layer. The first value in the model is 
# the number of nodes.  The input shape is needed for only the first layer, and only for the 
# number of columns (predictor variables),not the number of rows (observations). The final layer
# is the output layer, which has only one prediction value. The activation function 'relu'
# is short for rectified linear unit.  Negative values fed into the relu function return zero. 
# Postive values fed into relu return the same value.  Note: the slope of relu function is zero
# for negative values and 1 for all other values.

# Import necessary modules
import keras
from keras.layers import Dense
from keras.models import Sequential

# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]

# Set up the model: model
model = Sequential()

# Add the first layer
model.add(Dense(50, activation='relu', input_shape=(n_cols,)))

# Add the second layer
model.add(Dense(32, activation = 'relu'))

# Add the output layer
model.add(Dense(1))







# Compiling the model

# Step 2 of the process is to compile the model.  Compiling  the model
# sets up the problem for efficient backpropogation.  In Keras, this involves
# selecting an optimization function that will control the learning rate, and a loss 
# function (which we use the slope of during gradient descent).  Here we 
# repeat the steps from above and add the compile step

# ADAM is a robust, go-to optimization method for most problems
# See https://keras.io/optimizers/#adam for documention on adam optimization
# See https://arxiv.org/abs/1412.6980v8 for the original paper

# Import necessary modules
import keras
from keras.layers import Dense
from keras.models import Sequential

# Specify the model
n_cols = predictors.shape[1]
model = Sequential()
model.add(Dense(50, activation='relu', input_shape = (n_cols,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Verify that model contains information from compiling
print("Loss function: " + model.loss)






### Fitting the model

# See above for previous steps, fitting is quite straight forward.
# Feed in a numpy matrix of predictor observations and variables and
# a numpy matrix of target values

# Import necessary modules
import keras
from keras.layers import Dense
from keras.models import Sequential

# Specify the model
n_cols = predictors.shape[1]
model = Sequential()
model.add(Dense(50, activation='relu', input_shape = (n_cols,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Fit the model
model.fit(predictors, target)





### Last steps in classification models

# Building the a.n.n. for classification requires some differences to 
# regression problems.  The target needs to be setup categorically (to_categorical 
# or one hot encoding could work), we use the softmax activation function
# and difference loss functions.  This example also uses 'sgd' (stochastic gradient
# descent) for optimizing

# Import necessary modules
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical

# Convert the target to categorical: target
target = to_categorical(df.survived)

# Set up the model
model = Sequential()

# Add the first layer
model.add(Dense(32,activation='relu', input_shape=(n_cols,)))

# Add the output layer
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model
model.fit(predictors, target)






### Making predictions

# Use the trained model to predict on new data.  The return value of 
# prediction is an array of probabilites that the passenger (1) perished
# or (2) survived. We're interested in the probabilities of survival, so 
# we've selected the second column of the arrays and saved it to the 
# variable (predicted_prob_true)

# Specify, compile, and fit the model
model = Sequential()
model.add(Dense(32, activation='relu', input_shape = (n_cols,)))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='sgd', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(predictors, target)

# Calculate predictions: predictions
predictions = model.predict(pred_data)

# Calculate predicted probability of survival: predicted_prob_true
predicted_prob_true = predictions[:,1]

# print predicted_prob_true
print(predicted_prob_true)

# Output:
   [ 0.71419144  0.99715388  1.          0.98716879  0.42880797  0.41993374
      0.12816767  0.664554    0.63532716  1.          0.48327175  0.97510242
      0.53130805  0.99999809  0.43752927  0.17634997  0.59050709  0.99998236
      0.24961966  0.99997413  1.          0.52088666  0.13235945  0.81609923
      1.          0.42093429  1.          1.          0.45627564  1.          0.9522959
      0.99988532  0.45558897  0.5349201   0.65219122  1.          0.62631118
      0.46427011  1.          0.99973077  0.59306842  0.81698     0.99986637
      0.3524766   0.69103783  0.29771155  1.          0.40880477  0.9999572   1.
      0.99999869  0.09472843  0.99211919  1.          0.98233193  0.77582765
      1.          0.97524345  0.98128271  0.45558897  0.59962463  0.78674716
      0.98049635  1.          0.83715487  0.63286662  0.86550164  1.
      0.53568262  0.98182136  0.48386717  0.99999988  0.29644206  0.30515665
      0.99510586  0.70010716  0.71678001  0.60835755  0.45739332  1.
      0.99936372  0.40339831  0.76545835  0.6525718   0.45430472  0.99730456
      0.80546325  0.99997914  0.9904626   0.99988401  0.41317677]













# CHAPTER 4 ######################################################################################














### Changing optimization parameters

# Here we use three different learning rates to understand the affect on the loss function,
# noting that the progression of the loss function and its value (smaller is better) indicates
# the appropriatness of the learning rate

# Note: getting the function code out of datacamp website with the following code:
# import inspect
# lines = inspect.getsourcelines(foo)
# print("".join(lines[0]))

# predictor and target variables, not complete, but for demonstration of this classification problem
predictors = [[3 22.0 1 ..., 0 0 1]
 [1 38.0 1 ..., 1 0 0]
 [3 26.0 0 ..., 0 0 1]
 ..., 
 [3 29.69911764705882 1 ..., 0 0 1]
 [1 26.0 0 ..., 1 0 0]
 [3 32.0 0 ..., 0 1 0]]

 target = [[ 1.  0.]
 [ 0.  1.]
 [ 0.  1.]
 ..., 
 [ 1.  0.]
 [ 0.  1.]
 [ 1.  0.]]

def get_new_model(input_shape = input_shape):
    model = Sequential()
    model.add(Dense(100, activation='relu', input_shape = input_shape))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    return(model)

# Import the SGD optimizer
from keras.optimizers import SGD

# Create list of learning rates: lr_to_test
lr_to_test = [0.000001,0.01,1]

# Loop over learning rates
for lr in lr_to_test:
    print('\n\nTesting model with learning rate: %f\n'%lr )
    
    # Build new model to test, unaffected by previous models
    model = get_new_model()
    
    # Create SGD optimizer with specified learning rate: my_optimizer
    my_optimizer = SGD(lr=lr)
    
    # Compile the model
    model.compile(optimizer=my_optimizer, loss='categorical_crossentropy')
    
    # Fit the model
    model.fit(predictors, target)

# Output
Testing model with learning rate: 0.000001

Epoch 1/10

 32/891 [>.............................] - ETA: 1s - loss: 3.2695
608/891 [===================>..........] - ETA: 0s - loss: 2.4315
891/891 [==============================] - 0s - loss: 2.5036     
Epoch 2/10

 32/891 [>.............................] - ETA: 0s - loss: 1.6469
608/891 [===================>..........] - ETA: 0s - loss: 2.5360
891/891 [==============================] - 0s - loss: 2.4717     
Epoch 3/10

 32/891 [>.............................] - ETA: 0s - loss: 2.9302
608/891 [===================>..........] - ETA: 0s - loss: 2.4824
891/891 [==============================] - 0s - loss: 2.4400     
Epoch 4/10

 32/891 [>.............................] - ETA: 0s - loss: 2.8320
608/891 [===================>..........] - ETA: 0s - loss: 2.5376
891/891 [==============================] - 0s - loss: 2.4082     
Epoch 5/10

 32/891 [>.............................] - ETA: 0s - loss: 1.9983
608/891 [===================>..........] - ETA: 0s - loss: 2.3308
891/891 [==============================] - 0s - loss: 2.3767     
Epoch 6/10

 32/891 [>.............................] - ETA: 0s - loss: 2.1723
608/891 [===================>..........] - ETA: 0s - loss: 2.3249
891/891 [==============================] - 0s - loss: 2.3453     
Epoch 7/10

 32/891 [>.............................] - ETA: 0s - loss: 2.3140
608/891 [===================>..........] - ETA: 0s - loss: 2.5136
891/891 [==============================] - 0s - loss: 2.3138     
Epoch 8/10

 32/891 [>.............................] - ETA: 0s - loss: 2.6729
608/891 [===================>..........] - ETA: 0s - loss: 2.2660
891/891 [==============================] - 0s - loss: 2.2824     
Epoch 9/10

 32/891 [>.............................] - ETA: 0s - loss: 2.2261
384/891 [===========>..................] - ETA: 0s - loss: 2.3612
768/891 [========================>.....] - ETA: 0s - loss: 2.3095
891/891 [==============================] - 0s - loss: 2.2513     
Epoch 10/10

 32/891 [>.............................] - ETA: 0s - loss: 3.1438
416/891 [=============>................] - ETA: 0s - loss: 2.2521
832/891 [===========================>..] - ETA: 0s - loss: 2.2150
891/891 [==============================] - 0s - loss: 2.2201     


Testing model with learning rate: 0.010000

Epoch 1/10

 32/891 [>.............................] - ETA: 1s - loss: 1.0910
576/891 [==================>...........] - ETA: 0s - loss: 1.8063
891/891 [==============================] - 0s - loss: 1.4069     
Epoch 2/10

 32/891 [>.............................] - ETA: 0s - loss: 2.1145
512/891 [================>.............] - ETA: 0s - loss: 0.7652
891/891 [==============================] - 0s - loss: 0.7036     
Epoch 3/10

 32/891 [>.............................] - ETA: 0s - loss: 0.5718
576/891 [==================>...........] - ETA: 0s - loss: 0.6731
891/891 [==============================] - 0s - loss: 0.6485     
Epoch 4/10

 32/891 [>.............................] - ETA: 0s - loss: 0.6152
608/891 [===================>..........] - ETA: 0s - loss: 0.6438
891/891 [==============================] - 0s - loss: 0.6213     
Epoch 5/10

 32/891 [>.............................] - ETA: 0s - loss: 0.5013
576/891 [==================>...........] - ETA: 0s - loss: 0.6143
891/891 [==============================] - 0s - loss: 0.6189     
Epoch 6/10

 32/891 [>.............................] - ETA: 0s - loss: 0.6652
576/891 [==================>...........] - ETA: 0s - loss: 0.6081
891/891 [==============================] - 0s - loss: 0.5999     
Epoch 7/10

 32/891 [>.............................] - ETA: 0s - loss: 0.6106
608/891 [===================>..........] - ETA: 0s - loss: 0.5913
891/891 [==============================] - 0s - loss: 0.5990     
Epoch 8/10

 32/891 [>.............................] - ETA: 0s - loss: 0.6119
576/891 [==================>...........] - ETA: 0s - loss: 0.5843
891/891 [==============================] - 0s - loss: 0.6046     
Epoch 9/10

 32/891 [>.............................] - ETA: 0s - loss: 0.6580
576/891 [==================>...........] - ETA: 0s - loss: 0.5896
891/891 [==============================] - 0s - loss: 0.5905     
Epoch 10/10

 32/891 [>.............................] - ETA: 0s - loss: 0.6478
576/891 [==================>...........] - ETA: 0s - loss: 0.5763
891/891 [==============================] - 0s - loss: 0.5831     


Testing model with learning rate: 1.000000

Epoch 1/10

 32/891 [>.............................] - ETA: 1s - loss: 1.0273
416/891 [=============>................] - ETA: 0s - loss: 5.5034
800/891 [=========================>....] - ETA: 0s - loss: 5.8033
891/891 [==============================] - 0s - loss: 5.9885     
Epoch 2/10

 32/891 [>.............................] - ETA: 0s - loss: 4.5332
384/891 [===========>..................] - ETA: 0s - loss: 6.0443
864/891 [============================>.] - ETA: 0s - loss: 6.1749
891/891 [==============================] - 0s - loss: 6.1867     
Epoch 3/10

 32/891 [>.............................] - ETA: 0s - loss: 7.0517
576/891 [==================>...........] - ETA: 0s - loss: 5.9883
891/891 [==============================] - 0s - loss: 6.1867     
Epoch 4/10

 32/891 [>.............................] - ETA: 0s - loss: 6.0443
576/891 [==================>...........] - ETA: 0s - loss: 6.3801
891/891 [==============================] - 0s - loss: 6.1867     
Epoch 5/10

 32/891 [>.............................] - ETA: 0s - loss: 9.0664
480/891 [===============>..............] - ETA: 0s - loss: 5.9100
891/891 [==============================] - 0s - loss: 6.1867     
Epoch 6/10

 32/891 [>.............................] - ETA: 0s - loss: 6.0443
576/891 [==================>...........] - ETA: 0s - loss: 6.1842
891/891 [==============================] - 0s - loss: 6.1867     
Epoch 7/10

 32/891 [>.............................] - ETA: 0s - loss: 5.0369
544/891 [=================>............] - ETA: 0s - loss: 6.3998
891/891 [==============================] - 0s - loss: 6.1867     
Epoch 8/10

 32/891 [>.............................] - ETA: 0s - loss: 5.0369
576/891 [==================>...........] - ETA: 0s - loss: 6.0723
891/891 [==============================] - 0s - loss: 6.1867     
Epoch 9/10

 32/891 [>.............................] - ETA: 0s - loss: 5.5406
576/891 [==================>...........] - ETA: 0s - loss: 6.1282
891/891 [==============================] - 0s - loss: 6.1867     
Epoch 10/10

 32/891 [>.............................] - ETA: 0s - loss: 5.5406
576/891 [==================>...........] - ETA: 0s - loss: 6.4640
891/891 [==============================] - 0s - loss: 6.1867










### Evaluating model accuracy on validation dataset

# To evaluate how well the model performs, we split the data during 
# the fit step using the validation_split argument.  During compiling 
# the model we also add the metrics keyword argument.  The training phase
# will now monitor both fit on the training data and accuracy on the 
# test data

# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]
input_shape = (n_cols,)

# Specify the model
model = Sequential()
model.add(Dense(100, activation='relu', input_shape = input_shape))
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

# Fit the model
hist = model.fit(predictors,target,validation_split=0.3)

# Output
  Train on 623 samples, validate on 268 samples
    Epoch 1/10
    
 32/623 [>.............................] - ETA: 1s - loss: 3.3028 - acc: 0.4062
544/623 [=========================>....] - ETA: 0s - loss: 1.3419 - acc: 0.5901
623/623 [==============================] - 0s - loss: 1.3118 - acc: 0.6003 - val_loss: 0.6823 - val_acc: 0.7201
    Epoch 2/10
    
 32/623 [>.............................] - ETA: 0s - loss: 0.6880 - acc: 0.7188
416/623 [===================>..........] - ETA: 0s - loss: 0.8392 - acc: 0.5721
623/623 [==============================] - 0s - loss: 0.8817 - acc: 0.5682 - val_loss: 1.1047 - val_acc: 0.6418
    Epoch 3/10
    
 32/623 [>.............................] - ETA: 0s - loss: 1.0348 - acc: 0.5938
576/623 [==========================>...] - ETA: 0s - loss: 0.8017 - acc: 0.6215
623/623 [==============================] - 0s - loss: 0.7978 - acc: 0.6244 - val_loss: 0.8579 - val_acc: 0.6381
    Epoch 4/10
    
 32/623 [>.............................] - ETA: 0s - loss: 0.6463 - acc: 0.6875
544/623 [=========================>....] - ETA: 0s - loss: 0.7703 - acc: 0.6507
623/623 [==============================] - 0s - loss: 0.7529 - acc: 0.6501 - val_loss: 0.6875 - val_acc: 0.7015
    Epoch 5/10
    
 32/623 [>.............................] - ETA: 0s - loss: 0.6847 - acc: 0.6250
544/623 [=========================>....] - ETA: 0s - loss: 0.6503 - acc: 0.6710
623/623 [==============================] - 0s - loss: 0.6780 - acc: 0.6421 - val_loss: 0.5921 - val_acc: 0.7201
    Epoch 6/10
    
 32/623 [>.............................] - ETA: 0s - loss: 0.5701 - acc: 0.6875
512/623 [=======================>......] - ETA: 0s - loss: 0.6719 - acc: 0.6445
623/623 [==============================] - 0s - loss: 0.6597 - acc: 0.6517 - val_loss: 0.5287 - val_acc: 0.7463
    Epoch 7/10
    
 32/623 [>.............................] - ETA: 0s - loss: 0.5635 - acc: 0.7500
544/623 [=========================>....] - ETA: 0s - loss: 0.6135 - acc: 0.6728
623/623 [==============================] - 0s - loss: 0.6011 - acc: 0.6806 - val_loss: 0.5111 - val_acc: 0.7201
    Epoch 8/10
    
 32/623 [>.............................] - ETA: 0s - loss: 0.5942 - acc: 0.7500
512/623 [=======================>......] - ETA: 0s - loss: 0.5826 - acc: 0.6973
623/623 [==============================] - 0s - loss: 0.5911 - acc: 0.6870 - val_loss: 0.5254 - val_acc: 0.7649
    Epoch 9/10
    
 32/623 [>.............................] - ETA: 0s - loss: 0.5621 - acc: 0.7188
512/623 [=======================>......] - ETA: 0s - loss: 0.6695 - acc: 0.6543
623/623 [==============================] - 0s - loss: 0.6742 - acc: 0.6565 - val_loss: 0.5625 - val_acc: 0.6940
    Epoch 10/10
    
 32/623 [>.............................] - ETA: 0s - loss: 0.4816 - acc: 0.8125
512/623 [=======================>......] - ETA: 0s - loss: 0.6238 - acc: 0.6875
623/623 [==============================] - 0s - loss: 0.6204 - acc: 0.6854 - val_loss: 0.5388 - val_acc: 0.7351









### Early stopping: Optimizing the optimization

# Rather than relying on the default 10 epochs to train, we can specify
# an early stopping monitor based on a patience argument.  The patience 
# argument specifies the number of epochs with no improvement before stopping.
# Passing this monitor to the callbacks argument of the fit method, we can
# also specify more epochs than the default 10, knowing that training will
# stop when a decent solution is reached.  The course suggests that models rarely improve
# after 2-3 epochs of no further improvement

# Import EarlyStopping
from keras.callbacks import EarlyStopping

# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]
input_shape = (n_cols,)

# Specify the model
model = Sequential()
model.add(Dense(100, activation='relu', input_shape = input_shape))
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

# Define early_stopping_monitor
early_stopping_monitor = EarlyStopping(patience=2)

# Fit the model
model.fit(predictors,target,validation_split=0.3,epochs=30,callbacks=[early_stopping_monitor])

# Output
    Train on 623 samples, validate on 268 samples
    Epoch 1/30
    
 32/623 [>.............................] - ETA: 1s - loss: 5.6563 - acc: 0.4688
544/623 [=========================>....] - ETA: 0s - loss: 1.6279 - acc: 0.5478
623/623 [==============================] - 0s - loss: 1.6365 - acc: 0.5650 - val_loss: 1.0830 - val_acc: 0.6679
    Epoch 2/30
    
 32/623 [>.............................] - ETA: 0s - loss: 1.8361 - acc: 0.4688
544/623 [=========================>....] - ETA: 0s - loss: 0.8486 - acc: 0.5956
623/623 [==============================] - 0s - loss: 0.8349 - acc: 0.6051 - val_loss: 0.5687 - val_acc: 0.7313
    Epoch 3/30
    
 32/623 [>.............................] - ETA: 0s - loss: 0.8445 - acc: 0.6562
544/623 [=========================>....] - ETA: 0s - loss: 0.6951 - acc: 0.6526
623/623 [==============================] - 0s - loss: 0.7150 - acc: 0.6565 - val_loss: 0.5295 - val_acc: 0.7537
    Epoch 4/30
    
 32/623 [>.............................] - ETA: 0s - loss: 1.0094 - acc: 0.6250
576/623 [==========================>...] - ETA: 0s - loss: 0.6673 - acc: 0.6771
623/623 [==============================] - 0s - loss: 0.6781 - acc: 0.6726 - val_loss: 0.5257 - val_acc: 0.7276
    Epoch 5/30
    
 32/623 [>.............................] - ETA: 0s - loss: 0.5502 - acc: 0.7188
576/623 [==========================>...] - ETA: 0s - loss: 0.6730 - acc: 0.6528
623/623 [==============================] - 0s - loss: 0.6801 - acc: 0.6517 - val_loss: 0.6853 - val_acc: 0.6754
    Epoch 6/30
    
 32/623 [>.............................] - ETA: 0s - loss: 0.4724 - acc: 0.7812
576/623 [==========================>...] - ETA: 0s - loss: 0.6313 - acc: 0.7066
623/623 [==============================] - 0s - loss: 0.6278 - acc: 0.7030 - val_loss: 0.5931 - val_acc: 0.7015
    Epoch 7/30
    
 32/623 [>.............................] - ETA: 0s - loss: 0.6612 - acc: 0.6562
512/623 [=======================>......] - ETA: 0s - loss: 0.6272 - acc: 0.7012
623/623 [==============================] - 0s - loss: 0.6421 - acc: 0.6982 - val_loss: 0.6888 - val_acc: 0.6455







### Experimenting with wider networks

# here we compare two models, one with 10 nodes in two hidden layers
# and one with 100 nodes in two hidden layers.  The results are plotted on
# a line graph as opposed to the 'verbose' output from earlier examples.
# The wider network proves to end with a better validation accuracy. See svg for plot

# Define early_stopping_monitor
early_stopping_monitor = EarlyStopping(patience=2)

# Create the new model: model_2
model_2 = Sequential()

# Add the first and second layers
model_2.add(Dense(100, activation='relu', input_shape=input_shape))
model_2.add(Dense(100, activation='relu', input_shape=input_shape))

# Add the output layer
model_2.add(Dense(2, activation='softmax'))

# Compile model_2
model_2.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])

# Fit model_1
model_1_training = model_1.fit(predictors, target, epochs=15, validation_split=0.2, callbacks=[early_stopping_monitor], verbose=False)

# Fit model_2
model_2_training = model_2.fit(predictors, target, epochs=15, validation_split=0.2, callbacks=[early_stopping_monitor], verbose=False)

# Create the plot
plt.plot(model_1_training.history['val_loss'], 'r', model_2_training.history['val_loss'], 'b')
plt.xlabel('Epochs')
plt.ylabel('Validation score')
plt.show()






### Adding layers to a network

# Same as above, only we can add more layers and compare results
# to a baseline model. See svg which indicates improvement, but depends on 
# when the training ends

# The input shape to use in the first hidden layer
input_shape = (n_cols,)

# Create the new model: model_2
model_2 = Sequential()

# Add the first, second, and third hidden layers
model_2.add(Dense(50,activation='relu',input_shape=input_shape))
model_2.add(Dense(50,activation='relu'))
model_2.add(Dense(50,activation='relu'))

# Add the output layer
model_2.add(Dense(2,activation='softmax'))

# Compile model_2
model_2.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

# Fit model 1
model_1_training = model_1.fit(predictors, target, epochs=20, validation_split=0.4, callbacks=[early_stopping_monitor], verbose=False)

# Fit model 2
model_2_training = model_2.fit(predictors, target, epochs=20, validation_split=0.4, callbacks=[early_stopping_monitor], verbose=False)

# Create the plot
plt.plot(model_1_training.history['val_loss'], 'r', model_2_training.history['val_loss'], 'b')
plt.xlabel('Epochs')
plt.ylabel('Validation score')
plt.show()







### Building your own digit recognition model

# A subset of the MNIST dataset is used here and has been reshaped
# from the 28x28 pixel image to a 1x784 array, each of which
# has a value between 0 and 1 representing darkness. Even with a subset of the
# data, this model achieves nearly 90% accuracy

X=array([[ 0.,  0.,  0., ...,  0.,  0.,  0.],
       [ 0.,  0.,  0., ...,  0.,  0.,  0.],
       [ 0.,  0.,  0., ...,  0.,  0.,  0.],
       ..., 
       [ 0.,  0.,  0., ...,  0.,  0.,  0.],
       [ 0.,  0.,  0., ...,  0.,  0.,  0.],
       [ 0.,  0.,  0., ...,  0.,  0.,  0.]], dtype=float32)
y=array([[ 0.,  0.,  0., ...,  0.,  0.,  0.],
       [ 0.,  1.,  0., ...,  0.,  0.,  0.],
       [ 0.,  0.,  0., ...,  0.,  0.,  0.],
       ..., 
       [ 0.,  0.,  0., ...,  0.,  0.,  0.],
       [ 0.,  0.,  0., ...,  0.,  1.,  0.],
       [ 0.,  1.,  0., ...,  0.,  0.,  0.]])

# Create the model: model
model = Sequential()

# Add the first hidden layer
model.add(Dense(50,activation='relu',input_shape=(784,)))

# Add the second hidden layer
model.add(Dense(50,activation='relu'))

# Add the output layer
model.add(Dense(10,activation='softmax'))

# Compile the model
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

# Fit the model
model.fit(X,y,validation_split=0.3)

# Output
    Train on 1750 samples, validate on 750 samples
    Epoch 1/10
    
  32/1750 [..............................] - ETA: 4s - loss: 2.3101 - acc: 0.0312
 288/1750 [===>..........................] - ETA: 0s - loss: 2.1996 - acc: 0.1632
 576/1750 [========>.....................] - ETA: 0s - loss: 2.0803 - acc: 0.2917
 864/1750 [=============>................] - ETA: 0s - loss: 1.9385 - acc: 0.4051
1184/1750 [===================>..........] - ETA: 0s - loss: 1.7873 - acc: 0.4882
1472/1750 [========================>.....] - ETA: 0s - loss: 1.6462 - acc: 0.5326
1750/1750 [==============================] - 0s - loss: 1.5235 - acc: 0.5686 - val_loss: 0.8475 - val_acc: 0.7853
    Epoch 2/10
    
  32/1750 [..............................] - ETA: 0s - loss: 0.7161 - acc: 0.8438
 224/1750 [==>...........................] - ETA: 0s - loss: 0.7550 - acc: 0.7679
 384/1750 [=====>........................] - ETA: 0s - loss: 0.7558 - acc: 0.7812
 576/1750 [========>.....................] - ETA: 0s - loss: 0.7088 - acc: 0.8021
 800/1750 [============>.................] - ETA: 0s - loss: 0.6810 - acc: 0.8175
1024/1750 [================>.............] - ETA: 0s - loss: 0.6629 - acc: 0.8232
1184/1750 [===================>..........] - ETA: 0s - loss: 0.6426 - acc: 0.8252
1408/1750 [=======================>......] - ETA: 0s - loss: 0.6197 - acc: 0.8338
1632/1750 [==========================>...] - ETA: 0s - loss: 0.6090 - acc: 0.8339
1750/1750 [==============================] - 0s - loss: 0.6102 - acc: 0.8331 - val_loss: 0.5254 - val_acc: 0.8507
    Epoch 3/10
    
  32/1750 [..............................] - ETA: 0s - loss: 0.4899 - acc: 0.8750
 256/1750 [===>..........................] - ETA: 0s - loss: 0.5447 - acc: 0.8516
 448/1750 [======>.......................] - ETA: 0s - loss: 0.5068 - acc: 0.8638
 640/1750 [=========>....................] - ETA: 0s - loss: 0.4721 - acc: 0.8812
 832/1750 [=============>................] - ETA: 0s - loss: 0.4663 - acc: 0.8750
1024/1750 [================>.............] - ETA: 0s - loss: 0.4493 - acc: 0.8828
1280/1750 [====================>.........] - ETA: 0s - loss: 0.4265 - acc: 0.8875
1536/1750 [=========================>....] - ETA: 0s - loss: 0.4090 - acc: 0.8939
1750/1750 [==============================] - 0s - loss: 0.4041 - acc: 0.8937 - val_loss: 0.4329 - val_acc: 0.8773
    Epoch 4/10
    
  32/1750 [..............................] - ETA: 0s - loss: 0.2996 - acc: 0.9062
 320/1750 [====>.........................] - ETA: 0s - loss: 0.2938 - acc: 0.9281
 544/1750 [========>.....................] - ETA: 0s - loss: 0.2980 - acc: 0.9246
 768/1750 [============>.................] - ETA: 0s - loss: 0.2985 - acc: 0.9245
 992/1750 [================>.............] - ETA: 0s - loss: 0.3024 - acc: 0.9264
1248/1750 [====================>.........] - ETA: 0s - loss: 0.3276 - acc: 0.9191
1440/1750 [=======================>......] - ETA: 0s - loss: 0.3289 - acc: 0.9187
1632/1750 [==========================>...] - ETA: 0s - loss: 0.3273 - acc: 0.9161
1750/1750 [==============================] - 0s - loss: 0.3243 - acc: 0.9171 - val_loss: 0.4290 - val_acc: 0.8813
    Epoch 5/10
    
  32/1750 [..............................] - ETA: 0s - loss: 0.3113 - acc: 0.9062
 224/1750 [==>...........................] - ETA: 0s - loss: 0.2371 - acc: 0.9464
 512/1750 [=======>......................] - ETA: 0s - loss: 0.2801 - acc: 0.9238
 736/1750 [===========>..................] - ETA: 0s - loss: 0.2847 - acc: 0.9198
1024/1750 [================>.............] - ETA: 0s - loss: 0.2600 - acc: 0.9307
1280/1750 [====================>.........] - ETA: 0s - loss: 0.2474 - acc: 0.9359
1504/1750 [========================>.....] - ETA: 0s - loss: 0.2592 - acc: 0.9328
1750/1750 [==============================] - 0s - loss: 0.2636 - acc: 0.9297 - val_loss: 0.3708 - val_acc: 0.8933
    Epoch 6/10
    
  32/1750 [..............................] - ETA: 0s - loss: 0.2828 - acc: 0.9062
 352/1750 [=====>........................] - ETA: 0s - loss: 0.2386 - acc: 0.9403
 672/1750 [==========>...................] - ETA: 0s - loss: 0.2255 - acc: 0.9464
 928/1750 [==============>...............] - ETA: 0s - loss: 0.2164 - acc: 0.9429
1152/1750 [==================>...........] - ETA: 0s - loss: 0.2185 - acc: 0.9410
1440/1750 [=======================>......] - ETA: 0s - loss: 0.2236 - acc: 0.9396
1728/1750 [============================>.] - ETA: 0s - loss: 0.2207 - acc: 0.9421
1750/1750 [==============================] - 0s - loss: 0.2195 - acc: 0.9423 - val_loss: 0.3694 - val_acc: 0.8893
    Epoch 7/10
    
  32/1750 [..............................] - ETA: 0s - loss: 0.0638 - acc: 1.0000
 320/1750 [====>.........................] - ETA: 0s - loss: 0.1543 - acc: 0.9781
 608/1750 [=========>....................] - ETA: 0s - loss: 0.1734 - acc: 0.9622
 832/1750 [=============>................] - ETA: 0s - loss: 0.1732 - acc: 0.9627
1056/1750 [=================>............] - ETA: 0s - loss: 0.1695 - acc: 0.9631
1280/1750 [====================>.........] - ETA: 0s - loss: 0.1707 - acc: 0.9633
1504/1750 [========================>.....] - ETA: 0s - loss: 0.1816 - acc: 0.9581
1664/1750 [===========================>..] - ETA: 0s - loss: 0.1767 - acc: 0.9603
1750/1750 [==============================] - 0s - loss: 0.1824 - acc: 0.9577 - val_loss: 0.3587 - val_acc: 0.8867
    Epoch 8/10
    
  32/1750 [..............................] - ETA: 0s - loss: 0.1277 - acc: 0.9688
 384/1750 [=====>........................] - ETA: 0s - loss: 0.1368 - acc: 0.9714
 704/1750 [===========>..................] - ETA: 0s - loss: 0.1406 - acc: 0.9659
 928/1750 [==============>...............] - ETA: 0s - loss: 0.1450 - acc: 0.9644
1280/1750 [====================>.........] - ETA: 0s - loss: 0.1559 - acc: 0.9586
1632/1750 [==========================>...] - ETA: 0s - loss: 0.1566 - acc: 0.9589
1750/1750 [==============================] - 0s - loss: 0.1590 - acc: 0.9589 - val_loss: 0.3489 - val_acc: 0.8987
    Epoch 9/10
    
  32/1750 [..............................] - ETA: 0s - loss: 0.1584 - acc: 0.9688
 384/1750 [=====>........................] - ETA: 0s - loss: 0.1083 - acc: 0.9844
 768/1750 [============>.................] - ETA: 0s - loss: 0.1290 - acc: 0.9727
1120/1750 [==================>...........] - ETA: 0s - loss: 0.1341 - acc: 0.9750
1472/1750 [========================>.....] - ETA: 0s - loss: 0.1341 - acc: 0.9715
1750/1750 [==============================] - 0s - loss: 0.1338 - acc: 0.9726 - val_loss: 0.3463 - val_acc: 0.8907
    Epoch 10/10
    
  32/1750 [..............................] - ETA: 0s - loss: 0.0648 - acc: 1.0000
 416/1750 [======>.......................] - ETA: 0s - loss: 0.1005 - acc: 0.9832
 832/1750 [=============>................] - ETA: 0s - loss: 0.1180 - acc: 0.9796
1184/1750 [===================>..........] - ETA: 0s - loss: 0.1176 - acc: 0.9780
1568/1750 [=========================>....] - ETA: 0s - loss: 0.1121 - acc: 0.9777
1750/1750 [==============================] - 0s - loss: 0.1107 - acc: 0.9771 - val_loss: 0.3486 - val_acc: 0.8867
