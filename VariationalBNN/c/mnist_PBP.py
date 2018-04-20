import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import math

import numpy as np

import sys
sys.path.append('PBP_net/')
import PBP_net
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.functional as F
import glob

np.random.seed(1)

# We load the boston housing dataset


# We obtain the features and the targets
train = datasets.MNIST('./data', train=True, transform=transforms.Compose([transforms.ToTensor()]), download=False)
test = datasets.MNIST('./dataTest', train=False, transform=transforms.Compose([transforms.ToTensor()]), download=False)

# We create the train and test sets with 90% and 10% of the data

X_train = train.train_data.view(-1, 28 * 28).numpy()
y_train = train.train_labels.numpy()
X_test = test.test_data.view(-1, 28 * 28).numpy()
y_test = test.test_labels.numpy()

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# We construct the network with one hidden layer with two-hidden layers
# with 50 neurons in each one and normalizing the training features to have
# zero mean and unit standard deviation in the trainig set.

n_hidden_units = 50
#net = PBP_net.load_PBP_net_from_file('pbp_network')
#net = PBP_net.PBP_net(X_train, y_train,
#    [n_hidden_units, n_hidden_units,  n_hidden_units, n_hidden_units, n_hidden_units ], normalize = True, n_epochs = 40)

# We make predictions for the test set
#for i in range(0, 100):
net.save_to_file('pbp_network')
outputs = np.zeros((y_test.shape[0], 10))
for j in range(0, 20):
    net.sample_weights()
    
    m, v, v_noise = net.predict(X_test)
    m = net.predict_deterministic(X_test)

    labels = np.rint(m)
    #what is good programming?
    labels[labels > 9.0] = 9 
    labels[labels < 0.0] = 0

    for k, l in enumerate(labels):
        outputs[k,int(l) ] += 1

winners = np.argmax(outputs, axis=1)
print(winners)

print(np.count_nonzero(y_test == np.int32(winners)) / float(y_test.shape[0]))




# We compute the test log-likelihood

#test_ll = np.mean(-0.5 * np.log(2 * math.pi * (v + v_noise)) - \
#    0.5 * (y_test - m)**2 / (v + v_noise))

#print test_ll
