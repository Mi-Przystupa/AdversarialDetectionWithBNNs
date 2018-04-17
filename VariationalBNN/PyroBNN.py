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


class BNN(nn.Module):
    def __init__(self, p, outputs=10):
        super(BNN, self).__init__()
        self.linear = nn.Linear(p, outputs)
        self.outputs = outputs

    def forward(self, x):
        #print(self.linear(x))
        x = self.linear(x)
        return F.softmax(x, dim=1)


bnn = BNN(p, 10)
bnn = bnn


def model(data):
    x_data = data[0]
    y_data = data[1]
    mu, sigma = Variable(torch.zeros(10, p)), Variable(10 * torch.ones(10, p))
    bias_mu, bias_sigma = Variable(torch.zeros(10)), Variable(10 * torch.ones(10))

    w_prior, b_prior = Normal(mu, sigma), Normal(bias_mu, bias_sigma)
    priors = {'linear.weight': w_prior, 'linear.bias': b_prior}

    lifted_module = pyro.random_module("module", bnn, priors)
    lifted_bnn_model = lifted_module()

    # run regressor forward conditioned on data
    prediction = lifted_bnn_model(x_data).squeeze()
    pyro.sample("obs",
                Categorical(ps=prediction), obs=y_data)


softplus = torch.nn.Softplus()


def guide(data):

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
n_epochs = 200 # 50 # 100

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
    #loss = nn.CrossEntropyLoss()
    y_preds = Variable(torch.zeros(100, 1))
    for i in range(20):

        sampled_reg_model = guide(None)

        pred = sampled_reg_model(x_data)
        _, out = torch.max(pred, 1)
        acc = np.count_nonzero(np.squeeze(out.data.cpu().numpy()) == np.int32(y_data.data.numpy().ravel())) / float(test.test_labels.size()[0])
        print(acc)



    #print( "Loss: ", loss(y_preds, y_data.long()).data[0])


if __name__ == '__main__':
    main()
