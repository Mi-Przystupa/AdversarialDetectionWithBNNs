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
import torch.nn.functional as F
from torch.autograd import Variable
import glob
import Uncertainty


np.random.seed(1)

# We load the boston housing dataset


# We obtain the features and the targets
from torch.utils.data.dataset import TensorDataset
train = datasets.CIFAR10('./dataC', train=True, transform=transforms.Compose([transforms.ToTensor()]))#,download=True)
train.train_labels = torch.Tensor(train.train_labels)
test = datasets.CIFAR10('./dataTestC', train=False, transform=transforms.Compose([transforms.ToTensor()]))#,download=True)
test.test_labels = torch.Tensor(test.test_labels)
train.train_data = torch.from_numpy(np.load('data_cifar/training_vectors'))
test.test_data = torch.from_numpy(np.load('data_cifar/validation_vectors'))
# We create the train and test sets with 90% and 10% of the data

X_train = train.train_data.view(-1, 28 * 28).numpy()
y_train = train.train_labels.numpy()
#X_test = test.test_data.view(-1, 28 * 28).numpy()
#y_test = test.test_labels.numpy()

print(X_train.shape)
print(y_train.shape)
#print(X_test.shape)
#print(y_test.shape)

# We construct the network with one hidden layer with two-hidden layers
# with 50 neurons in each one and normalizing the training features to have
# zero mean and unit standard deviation in the trainig set.

n_hidden_units = 50
#net = PBP_net.load_PBP_net_from_file('pbp_network_cifar')
net = PBP_net.PBP_net(X_train, y_train,
    [n_hidden_units, n_hidden_units,  n_hidden_units, n_hidden_units, n_hidden_units ], normalize = True, n_epochs = 50)

# We make predictions for the test set
#for i in range(0, 100):
net.save_to_file('pbp_network_cifar')

def createProbabilityOfClasses(X_test, samples=10):
    outputs = np.zeros((X_test.shape[0], 10))
    X_test = X_test.data.cpu().numpy()
    for j in range(0, samples):
        net.sample_weights()
        #m, v, v_noise = net.predict(X_test)
        m = net.predict_deterministic(X_test)

        labels = np.rint(m)
        #what is good programming?
        labels[labels > 9.0] = 9 
        labels[labels < 0.0] = 0

        for k, l in enumerate(labels):
            outputs[k,int(l) ] += 1
    outputs = F.softmax(Variable(torch.from_numpy(outputs)), dim=1)
    return outputs


#print(np.count_nonzero(y_test == np.int32(winners)) / float(y_test.shape[0]))

######################################################################################################
#test model
outputs = 10
datasets = {'RegularImages_0.0': [test.test_data, test.test_labels]}

fgsm = glob.glob('fgsm/fgsm_cifar10_examples_x_10000_*')
fgsm_labels = test.test_labels 
for file in fgsm:
    parts = file.split('_')
    key = parts[0].split('/')[0] + '_' + parts[-1].split('.npy')[0]

    datasets[key] = [torch.from_numpy(np.load(file)), fgsm_labels]

#jsma = glob.glob('jsma/jsma_mnist_adv_x_10000*')
#jsma_labels = test.test_labels
#torch.from_numpy(np.argmax(np.load('jsma/jsma_mnist_adv_y_10000.npy'), axis=1))
#for file in jsma:
#    parts = file.split('_')
#    key = parts[0].split('/')[0] + '_' + parts[-1].split('.npy')[0]
#    datasets[key] = [torch.from_numpy(np.load(file)), jsma_labels]

gaussian = glob.glob('gaussian/cifar_gaussian_adv_x*')
gaussian_labels = test.test_labels 
for file in gaussian:
    parts = file.split('_')
    key = parts[0].split('/')[0] + '_' + parts[-1].split('.npy')[0]

    datasets[key] = [torch.from_numpy(np.load(file)), jsma_labels]



print(datasets.keys())
print('################################################################################')
accuracies = {}
for key, value in datasets.iteritems():
    print(key)
    parts = key.split('_')
    adversary_type = parts[0]
    epsilon = parts[1]
    data = value
    X, y = data[0].view(-1, 28 * 28), data[1]
    x_data, y_data = Variable(X.float().cuda()), Variable(y.cuda())
    T = 50

    accs = []
    samples = np.zeros((y_data.data.size()[0], T, outputs))
    for i in range(T):
        pred = createProbabilityOfClasses(x_data, samples=20)
        samples[:, i, :] = pred.data.cpu().numpy()
        _, out = torch.max(pred, 1)
        acc = np.count_nonzero(np.squeeze(out.data.cpu().numpy()) == np.int32(y_data.data.cpu().numpy().ravel())) / float(len(y_data.data.cpu().numpy()))
        accs.append(acc)

    variationRatio = []
    mutualInformation = []
    predictiveEntropy = []
    predictions = []

    for i in range(0, len(y_data)):
        entry = samples[i, :, :]
        variationRatio.append(Uncertainty.variation_ratio(entry))
        mutualInformation.append(Uncertainty.mutual_information(entry))
        predictiveEntropy.append(Uncertainty.predictive_entropy(entry))
        predictions.append(np.max(entry.mean(axis=0), axis=0))


    uncertainty={}
    uncertainty['varation_ratio']= np.array(variationRatio)
    uncertainty['predictive_entropy']= np.array(predictiveEntropy)
    uncertainty['mutual_information']= np.array(mutualInformation)
    predictions = np.array(predictions)

    Uncertainty.plot_uncertainty(uncertainty,predictions,adversarial_type=adversary_type,epsilon=float(epsilon), directory='Results_CIFAR_PBP')

    accs = np.array(accs)
    print('Accuracy mean: {}, Accuracy std: {}'.format(accs.mean(), accs.std()))
    accuracies[key] = {'mean': accs.mean(), 'std': accs.std()}

np.save('PBP_Accuracies_CIFAR', accuracies)

