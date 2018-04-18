import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import math

import numpy as np

import sys
sys.path.append('PBP_net/')
import PBP_net
from  tensorflow.examples.tutorials.mnist import input_data

np.random.seed(1)

# We load the boston housing dataset

data = np.loadtxt('boston_housing.txt')
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

# We obtain the features and the targets

X = mnist.train.images
y = mnist.train.labels

# We create the train and test sets with 90% and 10% of the data

X_train = mnist.train.images
y_train = mnist.train.labels 
X_test = mnist.test.images 
y_test = mnist.test.labels 

# We construct the network with one hidden layer with two-hidden layers
# with 50 neurons in each one and normalizing the training features to have
# zero mean and unit standard deviation in the trainig set.

n_hidden_units = 50
net = PBP_net.PBP_net(X_train, y_train,
    [ n_hidden_units, n_hidden_units ], normalize = True, n_epochs = 1)

# We make predictions for the test set

m, v, v_noise = net.predict(X_test)

# We compute the test RMSE

rmse = np.sqrt(np.mean((y_test - m)**2))

print rmse

# We compute the test log-likelihood

test_ll = np.mean(-0.5 * np.log(2 * math.pi * (v + v_noise)) - \
    0.5 * (y_test - m)**2 / (v + v_noise))

print test_ll
