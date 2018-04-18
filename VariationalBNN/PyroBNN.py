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

p = 28 * 28
hidden = 200
outputs = 10

class BNN(nn.Module):
    def __init__(self, p, outputs=10, hidden=200):
        super(BNN, self).__init__()
        self.linear = nn.Linear(p, hidden)
        self.linear2 = nn.Linear(hidden, outputs)
        #self.output = nn.Linear(hidden, outputs)

    def forward(self, x):
        #print(self.linear(x))
        x = self.linear(x)
        x = F.relu(x)
        x = self.linear2(x)
        #x = self.output(x)
        return F.softmax(x, dim=1)


bnn = BNN(p, 10, hidden=hidden)
bnn = bnn

#for p in bnn.named_parameters():
#    print(p[0])
#    print(p[1].data.shape())
def model(data):
    x_data = data[0]
    y_data = data[1]
    '''
    mu, sigma = Variable(torch.zeros(10, p)), Variable(10 * torch.ones(10, p))
    bias_mu, bias_sigma = Variable(torch.zeros(10)), Variable(10 * torch.ones(10))

    w_prior, b_prior = Normal(mu, sigma), Normal(bias_mu, bias_sigma)
    priors = {'linear.weight': w_prior, 'linear.bias': b_prior}
    '''
    mu1, sigma1 = Variable(torch.zeros(hidden, p)), Variable(10 * torch.ones(hidden, p))
    bias_mu1, bias_sigma1 = Variable(torch.zeros(hidden)), Variable(10 * torch.ones(hidden))
    mu2, sigma2 = Variable(torch.zeros(outputs, hidden)), Variable(10 * torch.ones(outputs, hidden))
    bias_mu2, bias_sigma2 = Variable(torch.zeros(outputs)), Variable(10 * torch.ones(outputs))

    w_prior1, b_prior1 = Normal(mu1, sigma1), Normal(bias_mu1, bias_sigma1)
    w_prior2, b_prior2 = Normal(mu2, sigma2), Normal(bias_mu2, bias_sigma2)

    priors = {'linear.weight': w_prior1, 'linear.bias': b_prior1, 'linear.weight2': w_prior2, 'linear.bias2': b_prior2}
    lifted_module = pyro.random_module("module", bnn, priors)
    lifted_bnn_model = lifted_module()

    # run regressor forward conditioned on data
    prediction = lifted_bnn_model(x_data).squeeze()
    pyro.sample("obs",
                Categorical(ps=prediction), obs=y_data)


softplus = torch.nn.Softplus()


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
    w_mu1 = Variable(torch.randn(hidden, p), requires_grad=True)
    w_log_sig1 = Variable(-3.0 * torch.ones(hidden, p) + 0.05 * torch.randn(hidden, p), requires_grad=True)
    b_mu1 = Variable(torch.randn(hidden), requires_grad=True)
    b_log_sig1 = Variable(-3.0 * torch.ones(hidden) + 0.05 * torch.randn(hidden),
                         requires_grad=True)
    w_mu2 = Variable(torch.randn(outputs, hidden), requires_grad=True)
    w_log_sig2 = Variable(-3.0 * torch.ones(outputs, hidden) + 0.05 * torch.randn(outputs, hidden), requires_grad=True)
    b_mu2 = Variable(torch.randn(outputs), requires_grad=True)
    b_log_sig2 = Variable(-3.0 * torch.ones(outputs) + 0.05 * torch.randn(outputs),
                         requires_grad=True)


    #register learnable param in the param store
    mw_param1 = pyro.param("guide_mean_weight", w_mu1)
    sw_param1 = softplus(pyro.param("guide_log_sigma_weight", w_log_sig1))
    mb_param1 = pyro.param("guide_mean_bias", b_mu1)
    sb_param1 = softplus(pyro.param("guide_log_sigma_bias", b_log_sig1))

    mw_param2 = pyro.param("guide_mean_weight2", w_mu2)
    sw_param2 = softplus(pyro.param("guide_log_sigma_weight2", w_log_sig2))
    mb_param2 = pyro.param("guide_mean_bias2", b_mu2)
    sb_param2 = softplus(pyro.param("guide_log_sigma_bias2", b_log_sig2))

    w_dist1, b_dist1 = Normal(mw_param1, sw_param1), Normal(mb_param1, sb_param1)
    w_dist2, b_dist2 = Normal(mw_param2, sw_param2), Normal(mb_param2, sb_param2)
    dists = {'linear.weight': w_dist1, 'linear.bias': b_dist1, 'linear.weight2': w_dist2, 'linear.bias2': b_dist2}
    lifted_module = pyro.random_module("module", bnn, dists)

    return lifted_module()


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
n_samples = 3

learning_rate = 0.0001
n_epochs = 10 # 50 # 100

batch_size = 100
n_batches = M / float(batch_size)


optim = Adam({"lr": 0.0001})
svi = SVI(model, guide, optim, loss="ELBO")
train_loader = utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
n_train_batches = int(train.train_labels.size()[0] / float(batch_size))


def main():
    pyro.clear_param_store()
    for j in range(n_epochs):
        loss = 0

        for data in train_loader:
            data[0] = Variable(data[0].view(-1, 28 * 28))
            data[1] = Variable(data[1].long())
            loss += svi.step(data)
        #if j % 100 == 0:
        print("[iteration %04d] loss: %.4f" % (j + 1, loss / float(n_train_batches * batch_size)))
    #for name in pyro.get_param_store().get_all_param_names():
    #    print("[%s]: %.3f" % (name, pyro.param(name).data.numpy()))

    data = [test.test_data, test.test_labels]
    X, y = data[0].view(-1, 28 * 28), data[1]
    print(X)
    print(y)

    x_data, y_data = Variable(X.float()), Variable(y)
    accs = []
    for i in range(20):

        sampled_reg_model = guide(None)

        pred = sampled_reg_model(x_data)
        _, out = torch.max(pred, 1)
        acc = np.count_nonzero(np.squeeze(out.data.cpu().numpy()) == np.int32(y_data.data.numpy().ravel())) / float(test.test_labels.size()[0])
        accs.append(acc)
        print(acc)
    accs = np.array(accs)
    print('Accuracy mean: {}, Accuracy std: {}'.format(accs.mean(), accs.std()))


    #print( "Loss: ", loss(y_preds, y_data.long()).data[0])


if __name__ == '__main__':
    main()
