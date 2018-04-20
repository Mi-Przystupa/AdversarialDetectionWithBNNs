import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Variable

import pyro
from pyro.distributions import Normal
from pyro.infer import SVI
from pyro.optim import Adam

N = 100
p = 1


def build_linear_dataset(N, noise_std=0.1):
    X = np.linspace(-6, 6, num=N)
    y = 3 * X + 1 + np.random.normal(0, noise_std, size=N)
    X, y = X.reshape((N,1 )), y.reshape((N,1))
    X,y = Variable(torch.Tensor(X)), Variable(torch.Tensor(y))
    return torch.cat((X , y), 1)

class RegressionModel(nn.Module):
    def __init__(self, p):
        super(RegressionModel, self).__init__()
        self.linear = nn.Linear(p, 1)

    def forward(self, x):
        return self.linear(x)

regression_model = RegressionModel(p)

loss_fn = torch.nn.MSELoss(size_average=False)
optim = torch.optim.Adam(regression_model.parameters(), lr=0.01)
num_iterations = 1000


def model(data):
    x_data = data[:,:-1]
    y_data = data[:, -1]

    mu, sigma = Variable(torch.zeros(1, p)), Variable(10 * torch.ones(1, p))
    bias_mu, bias_sigma = Variable(torch.zeros(1)), Variable(10 * torch.ones(1))

    w_prior, b_prior = Normal(mu, sigma), Normal(bias_mu, bias_sigma)
    priors = {'linear.weight' : w_prior, 'linear.bias': b_prior}

    lifted_module = pyro.random_module("module", regression_model, priors)
    lifted_reg_model = lifted_module()

    # run regressor forward conditioned on data
    prediction_mean = lifted_reg_model(x_data).squeeze()

    pyro.sample("obs",
                Normal(prediction_mean, Variable(0.1 * torch.ones(data.size(0)))), obs=y_data.squeeze())

softplus = torch.nn.Softplus()

def guide(data):

    w_mu = Variable(torch.randn(1, p), requires_grad=True)

    w_log_sig = Variable(-3.0 * torch.ones(1, p) + 0.05 * torch.randn(1, p), requires_grad=True)

    b_mu = Variable(torch.randn(1), requires_grad=True)
    b_log_sig = Variable(-3.0 * torch.ones(1) + 0.05 * torch.randn(1),
                         requires_grad=True)

    #register learnable param in the param store
    mw_param = pyro.param("guide_mean_weight", w_mu)
    sw_param = softplus(pyro.param("guide_log_sigma_weight", w_log_sig))
    mb_param = pyro.param("guide_mean_bias", b_mu)
    sb_param = softplus(pyro.param("guide_log_sigma_bias", b_log_sig))

    w_dist, b_dist = Normal(mw_param, sw_param), Normal(mb_param, sb_param)
    dists = {'linear.weight': w_dist, 'linear.bias': b_dist}
    lifted_module = pyro.random_module("module", regression_model, dists)

    return lifted_module()

optim = Adam({"lr": 0.01})
svi = SVI(model, guide, optim, loss="ELBO")

def main():
    pyro.clear_param_store()
    data = build_linear_dataset(N, p)
    for j in range(num_iterations):
        loss = svi.step(data)
        if j % 100 == 0:
            print("[iteration %04d] loss: %.4f" % (j + 1, loss / float(N)))
    for name in pyro.get_param_store().get_all_param_names():
        print("[%s]: %.3f" % (name, pyro.param(name).data.numpy()))

    X = np.linspace(6, 7, num= 20)
    y = 3 * X + 1
    X, y = X.reshape((20, 1)), y.reshape((20, 1))

    x_data, y_data = Variable(torch.Tensor(X)), Variable(torch.Tensor(y))
    loss = nn.MSELoss()
    y_preds = Variable(torch.zeros(20, 1))
    for i in range(20):

        sampled_reg_model = guide(None)

        y_preds = y_preds + sampled_reg_model(x_data)

    y_preds = y_preds / 20
    print( "Loss: ", loss(y_preds, y_data).data[0])
    '''
    data = build_linear_dataset(N, p)
    x_data = data[:,:-1]
    y_data = data[:, -1]
    for j in range(num_iterations):

        # run the model forward on the data
        y_pred = regression_model(x_data)
        # calculate the mse_loss
        loss = loss_fn(y_pred, y_data)
        # initialize gradients to zero
        optim.zero_grad()
        loss.backward()
        # take a gradient step
        optim.step() if (j + 1) % 50 == 0: print( "[iteration %04d] loss: %.4f" % (j + 1, loss.data[0]))
    print("Learned parameters:")

    for name, param in regression_model.named_parameters():
        print("%s: %.3f" % (name, param.data.numpy()))
    '''
if __name__ == '__main__':
    main()
