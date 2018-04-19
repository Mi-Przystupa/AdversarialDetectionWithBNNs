# Drawn from https://gist.github.com/rocknrollnerd/c5af642cf217971d93f499e8f70fcb72 (in Theano)
# This is implemented in PyTorch
# Author : Anirudh Vemula

import torch
import torch.nn as nn
from torch.autograd import Variable
import glob
import numpy as np

from sklearn import preprocessing
from torchvision import datasets
import Uncertainty


def log_gaussian(x, mu, sigma):
    #our p
    return float(-0.5 * np.log(2 * np.pi) - np.log(np.abs(sigma))) - (x - mu)**2 / (2 * sigma**2)

def gaussian(x, mu, sigma):
    scaling = 1.0 / np.sqrt(2.0 * np.pi * (sigma ** 2))
    bell = torch.exp(- (x - mu) ** 2 / (2.0 * sigma ** 2))

    return torch.mul(bell,scaling ) #scaling * bell

def scale_mixture_prior(x, sigma1, sigma2, pi):
    first_gaussian = pi * gaussian(x, 0., sigma1)
    second_gaussian = (1 - pi) * gaussian(x, 0., sigma2)

    return torch.log(first_gaussian + second_gaussian)


def log_gaussian_logsigma(x, mu, logsigma):
    #this...is identical to the above but is given np.log(sigma) as input I guess??
    return float(-0.5 * np.log(2 * np.pi)) - logsigma - (x - mu)**2 / (2 * torch.exp(logsigma)**2)


class MLPLayer(nn.Module):
    def __init__(self, n_input, n_output, sigma_prior):
        super(MLPLayer, self).__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.sigma_prior = sigma_prior
        self.W_mu = nn.Parameter(torch.Tensor(n_input, n_output).normal_(0, 0.01))
        self.W_logsigma = nn.Parameter(torch.Tensor(n_input, n_output).normal_(0, 0.01))
        self.b_mu = nn.Parameter(torch.Tensor(n_output).uniform_(-0.01, 0.01))
        self.b_logsigma = nn.Parameter(torch.Tensor(n_output).uniform_(-0.01, 0.01))
        self.lpw = 0
        self.lqw = 0

    def forward(self, X, infer=False):
        if infer:
            epsilon_W, epsilon_b = self.get_random()
            #W = torch.normal(self.W_mu, self.W_logsigma)
            #b = torch.normal(self.b_mu, self.b_logsigma)
            W = self.W_mu + torch.log(1 + torch.exp(self.W_logsigma)) * epsilon_W
            b = self.b_mu + torch.log(1 + torch.exp(self.b_logsigma)) * epsilon_b
            #output = torch.mm(X, self.W_mu) + self.b_mu.expand(X.size()[0], self.n_output)
            output = torch.mm(X, W) + b.expand(X.size()[0], self.n_output)
            return output

        epsilon_W, epsilon_b = self.get_random()
        W = self.W_mu + torch.log(1 + torch.exp(self.W_logsigma)) * epsilon_W
        b = self.b_mu + torch.log(1 + torch.exp(self.b_logsigma)) * epsilon_b
        output = torch.mm(X, W) + b.expand(X.size()[0], self.n_output)
        #self.lpw = log_gaussian(W, 0, self.sigma_prior).sum() + log_gaussian(b, 0, self.sigma_prior).sum()
        self.lpw  = scale_mixture_prior(W,self.sigma_prior, self.sigma_prior / 2., .1).sum() + scale_mixture_prior(b, self.sigma_prior, self.sigma_prior / 2., .1).sum()
        self.lqw = log_gaussian_logsigma(W, self.W_mu, self.W_logsigma).sum() + log_gaussian_logsigma(b, self.b_mu, self.b_logsigma).sum()
        return output

    def get_random(self):
        return Variable(torch.Tensor(self.n_input, self.n_output).normal_(0, self.sigma_prior).cuda()), Variable(torch.Tensor(self.n_output).normal_(0, self.sigma_prior).cuda())


class MLP(nn.Module):
    def __init__(self, n_input, sigma_prior):
        super(MLP, self).__init__()
        self.l1 = MLPLayer(n_input, 200, sigma_prior)
        self.l1_relu = nn.ReLU()
        self.l2 = MLPLayer(200, 200, sigma_prior)
        self.l2_relu = nn.ReLU()
        self.l3 = MLPLayer(200, 10, sigma_prior)
        self.l3_softmax = nn.Softmax()

    def forward(self, X, infer=False):
        output = self.l1_relu(self.l1(X, infer))
        output = self.l2_relu(self.l2(output, infer))
        output = self.l3_softmax(self.l3(output, infer))
        return output

    def get_lpw_lqw(self):
        lpw = self.l1.lpw + self.l2.lpw + self.l3.lpw
        lqw = self.l1.lqw + self.l2.lqw + self.l3.lqw
        return lpw, lqw


def forward_pass_samples(X, y):
    s_log_pw, s_log_qw, s_log_likelihood = 0., 0., 0.
    for _ in xrange(n_samples):
        output = net(X)
        sample_log_pw, sample_log_qw = net.get_lpw_lqw()
        sample_log_likelihood = log_gaussian(y, output, sigma_prior).sum()
        s_log_pw += sample_log_pw
        s_log_qw += sample_log_qw
        s_log_likelihood += sample_log_likelihood

    return s_log_pw/n_samples, s_log_qw/n_samples, s_log_likelihood/n_samples

# the criterion is 1/minibatch size * F(D, theta).
def criterion(l_pw, l_qw, l_likelihood):
    # so this is...seemingly correct
    return ((1./n_batches) * (l_qw - l_pw) - l_likelihood).sum() / float(batch_size)



train_target = np.float32(preprocessing.OneHotEncoder(sparse=False).fit_transform(train_target))

n_input = train_data.shape[1]
M = train_data.shape[0]
sigma_prior = float(np.exp(-3))
n_samples = 3
learning_rate = 0.0001
n_epochs = 100 # 50 # 100

# Initialize network
net = MLP(n_input, sigma_prior)

net = net.cuda()

# building the objective
# remember, we're evaluating by samples
log_pw, log_qw, log_likelihood = 0., 0., 0.
batch_size = 100
n_batches = M / float(batch_size)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

n_train_batches = int(train_data.shape[0] / float(batch_size))

for e in xrange(n_epochs):
    errs = []
    totalloss = 0.0
    for b in range(n_train_batches):
        net.zero_grad()
        X = Variable(torch.Tensor(train_data[b * batch_size: (b+1) * batch_size]).cuda())
        y = Variable(torch.Tensor(train_target[b * batch_size: (b+1) * batch_size]).cuda())
        #technically you're supposed to get n number of samples for each update...
        #so secretly there should be aloop around this where we calc the forward pas smultipel times
        log_pw, log_qw, log_likelihood = forward_pass_samples(X, y)
        #then we sum of thr esults in the criteron and do that business
        #F(D, theta)  = 1/n_batch (log_qw - log_pw) - logP(D | w), P(D|w) - softmax I believe...
        loss = criterion(log_pw, log_qw, log_likelihood)
        errs.append(loss.data.cpu().numpy())
        loss.backward()
        optimizer.step()
        totalloss += loss.data[0]

    #X = Variable(torch.Tensor(test_data).cuda(), volatile=True)
    #accs = []
    #for t in range(0, 10):
    #    pred = net(X, infer=True)
    #    _, out = torch.max(pred, 1)
    #    acc = np.count_nonzero(np.squeeze(out.data.cpu().numpy()) == np.int32(test_target.ravel())) / float(test_data.shape[0])
    #    accs.append(acc)
    #accs = np.array(accs)

    print 'epoch', e, 'loss', totalloss / n_train_batches#, 'acc_mean', accs.mean(), 'acc_std', accs.std()


######################################################################################################
#test model
outputs = 10
datasets = {'RegularImages_0.0': [test.test_data, test.test_labels]}

fgsm = glob.glob('fgsm/fgsm_mnist_adv_x_1000_*')
fgsm_labels  = torch.from_numpy(np.load('fgsm/fgsm_mnist_adv_y_1000.npy'))
for file in fgsm:
    parts = file.split('_')
    key = parts[0].split('/')[0] + '_' + parts[-1].split('.npy')[0]

    datasets[key] = [torch.from_numpy(np.load(file)), fgsm_labels]

jsma = glob.glob('jsma/jsma_mnist_adv_x_10000*')
jsma_labels = torch.from_numpy(np.load('jsma/jsma_mnist_adv_y_10000.npy'))
for file in jsma:
    parts = file.split('_')
    key = parts[0].split('/')[0] + '_' + parts[-1].split('.npy')[0]

    datasets[key] = [torch.from_numpy(np.load(file)), jsma_labels]
print(datasets.keys())
print('################################################################################')
for key, value in datasets.iteritems():
    print(key)
    parts = key.split('_')
    adversary_type = parts[0]
    epsilon = parts[1]
    print(epsilon)
    data = value
    X, y = data[0].view(-1, 28 * 28), data[1]
    x_data, y_data = Variable(X.float().cuda()), Variable(y.cuda())
    T = 100

    accs = []
    samples = np.zeros((y_data.data.size()[0], T, outputs))
    for i in range(T):
        pred = net(x_data,infer=True)
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

    Uncertainty.plot_uncertainty(uncertainty,predictions,adversarial_type=adversary_type,epsilon=float(epsilon))

    accs = np.array(accs)
    print('Accuracy mean: {}, Accuracy std: {}'.format(accs.mean(), accs.std()))

accs = []
for y in range(0, 50):
    pred = net(X,infer=True)
    _, out = torch.max(pred, 1)
    acc = np.count_nonzero(np.squeeze(out.data.cpu().numpy()) == np.int32(test_target.ravel())) / float(test_data.shape[0])
    accs.append(acc)
    print 'acc', acc

accs = np.array(accs)
print 'mean', accs.mean(), 'std', accs.std()
