import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils as utils
from torch.autograd import Variable
import numpy as np
import pyro
from pyro.distributions import Normal, Categorical
from pyro.infer import SVI
from pyro.optim import Adam

from sklearn import preprocessing
from torchvision import datasets, transforms
import time
import Uncertainty
import glob
p = 28 * 28
hidden = 1200
outputs = 10

class BNN(nn.Module):
    def __init__(self, p, outputs=10, hidden=200):
        super(BNN, self).__init__()
        self.linear = nn.Linear(p, hidden)
        self.linear2 = nn.Linear(hidden, hidden)
        self.linear3 = nn.Linear(hidden, outputs)
        #self.output = nn.Linear(hidden, outputs)

    def forward(self, x):
        #print(self.linear(x))
        x = self.linear(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        #x = self.output(x)
        return F.softmax(x, dim=1)


bnn = BNN(p, 10, hidden=hidden)
bnn = bnn.cuda()

for param in bnn.named_parameters():
    print(param[0])
    print(param[1].data.size())

mu1, sigma1 = Variable(torch.zeros(hidden, p).cuda()), Variable(10 * torch.ones(hidden, p).cuda())
bias_mu1, bias_sigma1 = Variable(torch.zeros(hidden).cuda()), Variable(10 * torch.ones(hidden).cuda())

mu2, sigma2 = Variable(torch.zeros(hidden, hidden).cuda()), Variable(10 * torch.ones(hidden, hidden).cuda())
bias_mu2, bias_sigma2 = Variable(torch.zeros(hidden).cuda()), Variable(10 * torch.ones(hidden).cuda())

mu3, sigma3 = Variable(torch.zeros(outputs, hidden).cuda()), Variable(10 * torch.ones(outputs, hidden).cuda())
bias_mu3, bias_sigma3 = Variable(torch.zeros(outputs).cuda()), Variable(10 * torch.ones(outputs).cuda())

def model(data):
    x_data = data[0]
    y_data = data[1]
    '''
    mu, sigma = Variable(torch.zeros(10, p)), Variable(10 * torch.ones(10, p))
    bias_mu, bias_sigma = Variable(torch.zeros(10)), Variable(10 * torch.ones(10))

    w_prior, b_prior = Normal(mu, sigma), Normal(bias_mu, bias_sigma)
    priors = {'linear.weight': w_prior, 'linear.bias': b_prior}
    '''

    w_prior1, b_prior1 = Normal(mu1, sigma1), Normal(bias_mu1, bias_sigma1)
    w_prior2, b_prior2 = Normal(mu2, sigma2), Normal(bias_mu2, bias_sigma2)
    w_prior3, b_prior3 = Normal(mu3, sigma3), Normal(bias_mu3, bias_sigma3)

    priors = {'linear.weight': w_prior1, 'linear.bias': b_prior1, 'linear2.weight': w_prior2, 'linear2.bias': b_prior2,
              'linear3.weight': w_prior3, 'linear3.bias': b_prior3}
    lifted_module = pyro.random_module("module", bnn, priors)
    lifted_bnn_model = lifted_module()

    # run regressor forward conditioned on data
    prediction = lifted_bnn_model(x_data).squeeze()
    pyro.sample("obs",
                Categorical(ps=prediction), obs=y_data)


softplus = torch.nn.Softplus()

w_mu1 = Variable(torch.randn(hidden, p).cuda(), requires_grad=True)
w_log_sig1 = Variable(-3.0 * torch.ones(hidden, p).cuda() + 0.05 * torch.randn(hidden, p).cuda(), requires_grad=True)
b_mu1 = Variable(torch.randn(hidden).cuda(), requires_grad=True)
b_log_sig1 = Variable(-3.0 * torch.ones(hidden).cuda() + 0.05 * torch.randn(hidden).cuda(),
                     requires_grad=True)
w_mu2 = Variable(torch.randn(hidden, hidden).cuda(), requires_grad=True)
w_log_sig2 = Variable(-3.0 * torch.ones(hidden, hidden).cuda() + 0.05 * torch.randn(hidden, hidden).cuda(), requires_grad=True)
b_mu2 = Variable(torch.randn(hidden).cuda(), requires_grad=True)
b_log_sig2 = Variable(-3.0 * torch.ones(hidden).cuda() + 0.05 * torch.randn(hidden).cuda(),
                     requires_grad=True)
w_mu3 = Variable(torch.randn(outputs, hidden).cuda(), requires_grad=True)
w_log_sig3 = Variable(-3.0 * torch.ones(outputs, hidden).cuda() + 0.05 * torch.randn(outputs, hidden).cuda(), requires_grad=True)
b_mu3 = Variable(torch.randn(outputs).cuda(), requires_grad=True)
b_log_sig3 = Variable(-3.0 * torch.ones(outputs).cuda() + 0.05 * torch.randn(outputs).cuda(),
                     requires_grad=True)


def guide(data):
    '''
    # define our theta values
    w_mu = Variable(torch.randn(10, p), requires_grad=True)
    w_log_sig = Variable(-3.0 * torch.ones(10,p) + 0.05 * torch.randn(1,p), requires_grad=True)
    b_mu = Variable(torch.randn(10), requires_grad=True)
    b_log_sig = Variable(-3.0 * torch.ones(10) + 0.05 * torch.randn(10),
                         requires_grad=True)

    #register learnable param in the param store
    mw_param = pyro.param("guide_mean_weight", w_mu)
    sw_param = softplus(pyro.param("guide_log_sigma_weight", w_log_sig))
    mb_param = pyro.param("guide_mean_bias", b_mu)
    sb_param = softplus(pyro.param("guide_log_sigma_bias", b_log_sig))

    w_dist, b_dist = Normal(mw_param, sw_param), Normal(mb_param, sb_param)
    dists = {'linear.weight': w_dist, 'linear.bias': b_dist}
    '''


    #register learnable param in the param store
    mw_param1 = pyro.param("guide_mean_weight", w_mu1)
    sw_param1 = softplus(pyro.param("guide_log_sigma_weight", w_log_sig1))
    mb_param1 = pyro.param("guide_mean_bias", b_mu1)
    sb_param1 = softplus(pyro.param("guide_log_sigma_bias", b_log_sig1))

    mw_param2 = pyro.param("guide_mean_weight2", w_mu2)
    sw_param2 = softplus(pyro.param("guide_log_sigma_weight2", w_log_sig2))
    mb_param2 = pyro.param("guide_mean_bias2", b_mu2)
    sb_param2 = softplus(pyro.param("guide_log_sigma_bias2", b_log_sig2))

    mw_param3 = pyro.param("guide_mean_weight3", w_mu3)
    sw_param3 = softplus(pyro.param("guide_log_sigma_weight3", w_log_sig3))
    mb_param3 = pyro.param("guide_mean_bias3", b_mu3)
    sb_param3 = softplus(pyro.param("guide_log_sigma_bias3", b_log_sig3))

    w_dist1, b_dist1 = Normal(mw_param1, sw_param1), Normal(mb_param1, sb_param1)
    w_dist2, b_dist2 = Normal(mw_param2, sw_param2), Normal(mb_param2, sb_param2)
    w_dist3, b_dist3 = Normal(mw_param3, sw_param3), Normal(mb_param3, sb_param3)
    dists = {'linear.weight': w_dist1, 'linear.bias': b_dist1, 'linear2.weight': w_dist2, 'linear2.bias': b_dist2, \
             'linear3.weight': w_dist3, 'linear3.bias': b_dist3}
    lifted_module = pyro.random_module("module", bnn, dists)

    return lifted_module()


train = datasets.MNIST('./data', train=True, transform=transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST('./dataTest', train=False, transform=transforms.Compose([transforms.ToTensor()]))

train_target = train.train_labels.unsqueeze(1).numpy()
test_target = test.test_labels.unsqueeze(1).numpy()
train_target = np.float32(preprocessing.OneHotEncoder(sparse=False).fit_transform(train_target))
test_target = np.float32(preprocessing.OneHotEncoder(sparse=False).fit_transform(test_target))

train.train_labels = torch.from_numpy(train_target)#.cuda()
#test.test_labels = torch.from_numpy(test_target)


n_input = 28 * 28
M = train.train_data.size()[0]
n_samples = 3

learning_rate = 0.0001
n_epochs = 20#20 # 50 # 100

batch_size = 256 * 2 * 2
n_batches = M / float(batch_size)


optim = Adam({"lr": 0.0001})
svi = SVI(model, guide, optim, loss="ELBO")
train_loader = utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
n_train_batches = int(train.train_labels.size()[0] / float(batch_size))


def main():
    pyro.clear_param_store()
    for j in range(n_epochs):
        loss = 0
        start = time.time()
        for data in train_loader:
            data[0] = Variable(data[0].view(-1, 28 * 28).cuda())
            data[1] = Variable(data[1].long().cuda())
            loss += svi.step(data)
        print(time.time() - start)
        #if j % 100 == 0:
        print("[iteration %04d] loss: %.4f" % (j + 1, loss / float(n_train_batches * batch_size)))
    #for name in pyro.get_param_store().get_all_param_names():
    #    print("[%s]: %.3f" % (name, pyro.param(name).data.numpy()))
    datasets =  {'RegularImages_0.0': [test.test_data, test.test_labels]}

    fgsm = glob.glob('fgsm/fgsm_mnist_adv_x_1000_*')
    fgsm_labels  = torch.from_numpy(np.argmax(np.load('fgsm/fgsm_mnist_adv_y_1000.npy'), axis=1))
    print(fgsm_labels)
    for file in fgsm:
        parts = file.split('_')
        key = parts[0].split('/')[0] + '_' + parts[-1].split('.npy')[0]

        datasets[key] = [torch.from_numpy(np.load(file)), fgsm_labels]

    jsma = glob.glob('jsma/jsma_mnist_adv_x_10000*')
    jsma_labels = torch.from_numpy(np.argmax(np.load('jsma/jsma_mnist_adv_y_10000.npy'), axis=1))
    for file in jsma:
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
        T = 100

        accs = []
        samples = np.zeros((y_data.data.size()[0], T, outputs))
        for i in range(T):
            sampled_model = guide(None)
            pred = sampled_model(x_data)
            samples[:, i, :] = pred.data.cpu().numpy()
            _, out = torch.max(pred, 1)

            acc = np.count_nonzero(np.squeeze(out.data.cpu().numpy()) == np.int32(y_data.data.cpu().numpy().ravel())) / float(y_data.data.size()[0])
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
        accuracies[key] = {'mean': accs.mean(), 'std': accs.std()}

    np.save('PyroBNN_accuracies', accuracies)


if __name__ == '__main__':
    main()
