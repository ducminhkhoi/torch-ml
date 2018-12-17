import torch 
from torch import nn
from torch.nn import functional as F
from torch.optim import SGD
from utils import get_blobs, get_spd
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt


class GMM:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        N, D = X.shape
        
        # step1: initialization
        indices = torch.randperm(N)[:self.n_components]

        mus = X[indices]
        sigmas = get_spd(self.n_components, D)
        pis = torch.ones(1, self.n_components) * 1/self.n_components

        normals = [MultivariateNormal(mus[i], sigmas[i]) for i in range(self.n_components)]
        p = torch.stack([normal.log_prob(X).exp() for normal in normals], 1)

        temp = pis * p

        prev_ll = temp.sum(1).log().sum()

        while True:
            print(prev_ll.item())
            # E step:
            gamma = temp / temp.sum(1, keepdim=True)

            # M step:
            Nk = gamma.sum(0, keepdim=True).t()
            
            mus = (gamma.t() @ X) / Nk

            diff = (X[:, None, :] - mus[None, :, :])[..., None]
            sigmas = (gamma[..., None, None] * diff @ diff.transpose(-2, -1)).sum(0) / Nk[..., None]

            pis = (Nk/N).t()

            # evaluate the log likelihood
            normals = [MultivariateNormal(mus[i], sigmas[i]) for i in range(self.n_components)]
            p = torch.stack([normal.log_prob(X).exp() for normal in normals], 1)

            temp = pis * p

            ll = temp.sum(1).log().sum()

            if abs(ll - prev_ll) < 1e-4:
                break
            else:
                prev_ll = ll

        self.mus = mus
        self.sigmas = sigmas
        self.pis = pis

    def predict(self, X):
        normals = [MultivariateNormal(self.mus[i], self.sigmas[i]) for i in range(self.n_components)]
        p = torch.stack([normal.log_prob(X).exp() for normal in normals], 1)

        temp = self.pis * p
        gamma = temp / temp.sum(1, keepdim=True)

        y = gamma.argmax(1)

        return y


class GMM2:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        N, D = X.shape
        
        # step1: initialization
        indices = torch.randperm(N)[:self.n_components]

        mus = nn.Parameter(X[indices])
        sigmas = nn.Parameter(get_spd(self.n_components, D))
        pis = nn.Parameter(torch.ones(1, self.n_components) * 1/self.n_components)

        optimizer = SGD([mus, sigmas, pis], lr=1e-2)

        prev_loss = 1e16

        with torch.enable_grad():
            while True:
                # E step: compute the negative log likelihood
                normals = [MultivariateNormal(mus[i], sigmas[i]) for i in range(self.n_components)]
                p = torch.stack([normal.log_prob(X).exp() for normal in normals], 1)

                temp = pis * p

                loss = -temp.sum(1).log().sum() / N

                print(loss.item())

                if abs(loss.item() - prev_loss) < 1e-4 or torch.isnan(loss):
                    break
                else:
                    prev_loss = loss.item()

                # M step: Optimize loss w.r.t mus, sigmas, pis
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # constraint pis to sum to 1 and gt 0
                pis.data = F.softmax(pis)

        self.mus = mus
        self.sigmas = sigmas
        self.pis = pis

    def predict(self, X):
        normals = [MultivariateNormal(self.mus[i], self.sigmas[i]) for i in range(self.n_components)]
        p = torch.stack([normal.log_prob(X).exp() for normal in normals], 1)

        temp = self.pis * p
        gamma = temp / temp.sum(1, keepdim=True)

        y = gamma.argmax(1)

        return y


class KNN:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        N, D = X.shape
        
        # step1: initialization
        indices = torch.randperm(N)[:self.n_components]
        mus = X[indices]

        dist = (X[:, None, :] - mus[None, :, :]).norm(2, -1)
        y = dist.argmin(1)
        prev_y = y

        while True:
            mus = torch.zeros_like(mus)
            for k in range(self.n_components):
                mus[k] = X[y==k].mean(0)

            dist = (X[:, None, :] - mus[None, :, :]).norm(2, -1)
            y = dist.argmin(1)

            diff = (prev_y != y).sum()
            print(diff)

            if diff == 0: # not changing membership
                break
            else:
                prev_y = y

        self.mus = mus

    def predict(self, X):
        dist = (X[:, None, :] - self.mus[None, :, :]).norm(2, -1)
        y = dist.argmin(1)

        return y


def visualize(inputs, targets, text='train_original'):
    classes = targets.unique()
    colors = ['r', 'g', 'b']

    _, ax = plt.subplots()

    for k, c in zip(classes, colors):
        input = inputs[targets==k]
        ax.scatter(input[:, 0], input[:, 1], c=c, label='Class {}'.format(k.item()))
    
    ax.legend()
    plt.savefig('figures/chapter7_{}.png'.format(text))


if __name__ == '__main__':
    X, y = get_blobs()

    model_name = 'GMM'

    model = globals()[model_name](3)

    with torch.no_grad():
        model.fit(X)

    y_predict = model.predict(X)

    visualize(X, y, text='gt')
    visualize(X, y_predict, text='{}'.format(model_name))


