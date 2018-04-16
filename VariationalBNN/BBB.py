# Drawn from https://gist.github.com/rocknrollnerd/c5af642cf217971d93f499e8f70fcb72 (in Theano)
# This is implemented in PyTorch
# Author : Anirudh Vemula

import torch
import torch.nn as nn
import torch.utils as utils
from torch.autograd import Variable
import numpy as np

from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
#from tensorflow.examples.tutorials.mnist import input_data
from torchvision import datasets, transforms
from torchviz import make_dot, make_dot_from_trace


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
#            W = torch.normal(self.W_mu, self.W_logsigma) # torch.log(1 + torch.exp(self.W_logsigma)) )
#            b = torch.normal(self.b_mu, self.b_logsigma) #torch.log(1 + torch.exp(self.b_logsigma)) )
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

class mnist:
    def __init__(self):
        #self.tf = input_data.read_data_sets("MNIST_data/",one_hot=False)
        train = datasets.MNIST('./data', train=True)#, download=True)
        test = datasets.MNIST('./dataTest', train=False)#,download=True)
        testdata = test.test_data.view(-1, 28 * 28)
        testlabel = test.test_labels
        traindata = train.train_data.view(-1, 28 * 28)
        trainlabel = train.train_labels
        self.data = torch.cat((traindata, testdata))
        self.target = torch.cat((trainlabel, testlabel))
mnist = mnist()
'''
N = 5000
data = np.float32(mnist.data[:]) / 255.
idx = np.random.choice(data.shape[0], N)
data = data[idx]
target = np.int32(mnist.target[idx]).reshape(N, 1)

train_idx, test_idx = train_test_split(np.array(range(N)), test_size=0.05)
train_data, test_data = data[train_idx], data[test_idx]
train_target, test_target = target[train_idx], target[test_idx]

train_target = np.float32(preprocessing.OneHotEncoder(sparse=False).fit_transform(train_target))
'''
train = datasets.MNIST('./data', train=True, transform=transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST('./dataTest', train=False, transform=transforms.Compose([transforms.ToTensor()]))

train_target = train.train_labels.unsqueeze(1).numpy()
test_target = test.test_labels.unsqueeze(1).numpy()
train_target = np.float32(preprocessing.OneHotEncoder(sparse=False).fit_transform(train_target))
test_target = np.float32(preprocessing.OneHotEncoder(sparse=False).fit_transform(test_target))

train.train_labels = torch.from_numpy(train_target)
#test.test_labels = torch.from_numpy(test_target)


n_input = 28 * 28
M = train.train_data.size()[0]
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

train_loader = utils.data.DataLoader(train, batch_size=batch_size, shuffle=True,
        )

n_train_batches = int(train.train_labels.size()[0] / float(batch_size))


for e in xrange(n_epochs):
    errs = []
    for b in train_loader:
        net.zero_grad()
        X = Variable(b[0].view(-1, 28 * 28).float().cuda())
        y = Variable(b[1].cuda())
        #technically you're supposed to get n number of samples for each update...
        #so secretly there should be aloop around this where we calc the forward pas smultipel times
        log_pw, log_qw, log_likelihood = forward_pass_samples(X, y)
        #then we sum of thr esults in the criteron and do that business
        #F(D, theta)  = 1/n_batch (log_qw - log_pw) - logP(D | w), P(D|w) - softmax I believe...
        loss = criterion(log_pw, log_qw, log_likelihood)
        make_dot(loss, params=dict(net.named_parameters()))
        errs.append(loss.data.cpu().numpy())
        loss.backward()
        optimizer.step()

    X = Variable(test.test_data.view(-1, 28 * 28).float().cuda(), volatile=True)
    accs = []
    for t in range(0, 10):
        pred = net(X, infer=True)
        _, out = torch.max(pred, 1)
        acc = np.count_nonzero(np.squeeze(out.data.cpu().numpy()) == np.int32(test.test_labels.numpy().ravel())) / float(test.test_labels.size()[0])
        accs.append(acc)
    accs = np.array(accs)

    print 'epoch', e, 'loss', np.mean(errs), 'acc_mean', accs.mean(), 'acc_std', accs.std()

accs = []
for y in range(0, 50):
    pred = net(X,infer=True)
    _, out = torch.max(pred, 1)
    acc = np.count_nonzero(np.squeeze(out.data.cpu().numpy()) == np.int32(test_target.ravel())) / float(test.test_labels.size()[0])
    accs.append(acc)
    print 'acc', acc

accs = np.array(accs)
print 'mean', accs.mean(), 'std', accs.std()
