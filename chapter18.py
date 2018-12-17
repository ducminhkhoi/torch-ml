import torch 
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.optim import SGD, Adam
import numpy as np

from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor

np.random.seed(1)

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt


def f(x):
    return x * np.sin(x)


def get_data(n_points=20):

    X = np.linspace(0.1, 9.9, n_points)
    X = np.atleast_2d(X).T

    # Observations and noise
    y = f(X).ravel()
    dy = 0.5 + 1.0 * np.random.random(y.shape)
    noise = np.random.normal(0, dy)
    y += noise

    x = np.atleast_2d(np.linspace(0, 10, 1000)).T

    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()[:, None]
    x = torch.from_numpy(x).float()

    return X, y, x, dy


class GaussianProcess:
    def __init__(self, noise=True):
        self.sigma_f = nn.Parameter(torch.ones(1))
        self.sigma_n = nn.Parameter(torch.ones(1))
        self.l = nn.Parameter(torch.ones(1))

    def fit(self, X, y):
        N, D = X.shape
        
        pdist = (X[:, None, :] - X[None, :, :]).norm(2, -1)**2

        optimizer = SGD([self.sigma_f, self.sigma_n, self.l], lr=1e-2)
        prev_loss = 1e16

        while True:
            K = self.sigma_f.pow(2) * torch.exp(-1/(2*self.l.pow(2)) * pdist) + \
                                self.sigma_n.pow(2) * torch.eye(N)

            K_inv = K.inverse()
            log_prob = -1/2*((y.t() @ K_inv @ y).squeeze() + K.det().log()) + N/2*np.log(2*np.pi)
            loss = -log_prob

            print(loss.item())

            if abs(loss.item() - prev_loss) < 1e-4 or torch.isnan(loss) \
                            or torch.isinf(loss) or loss.item() < 1:
                break
            else:
                prev_loss = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.X = X 
        self.y = y
        self.K_inv = K_inv

    def predict(self, X, return_std=False, return_cov=False):
        pdist = (X[:, None, :] - self.X[None, :, :]).norm(2, -1)**2
        k_star = self.sigma_f.pow(2) * torch.exp(-1/(2*self.l.pow(2)) * pdist)

        pdist = (X[:, None, :] - X[None, :, :]).norm(2, -1)**2
        k_star_star = self.sigma_f.pow(2) * torch.exp(-1/(2*self.l.pow(2)) * pdist)

        mean = k_star @ self.K_inv @ self.y

        cov, var = None, None

        if return_cov:
            cov = k_star_star - k_star @ self.K_inv @ k_star.t()

        if return_std:
            var = k_star_star.diag()
            var -= torch.einsum("ij,ij->i", k_star @ self.K_inv, k_star)
            var[var < 0] = 0

        return mean, var.sqrt(), cov


def visualize(X, x, y, y_pred, sigma, dy):
    X = X.data.numpy()
    x = x.data.numpy()
    y_pred = y_pred.data.numpy()
    sigma = sigma.data.numpy()

    plt.figure()
    plt.plot(x, f(x), 'r:', label=u'$f(x) = x\,\sin(x)$')
    plt.errorbar(X.ravel(), y, dy, fmt='r.', markersize=10, label=u'Observations')
    plt.plot(x, y_pred, 'b-', label=u'Prediction')
    plt.fill(np.concatenate([x, x[::-1]]),
            np.concatenate([y_pred - 1.9600 * sigma,
                            (y_pred + 1.9600 * sigma)[::-1]]),
            alpha=.5, fc='b', ec='None', label='95% confidence interval')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.ylim(-10, 20)
    plt.legend(loc='upper left')

    plt.savefig('figures/chapter10.png')


if __name__ == '__main__':
    X, y, x, dy = get_data(n_points=20)

    gp = GaussianProcess()

    gp.fit(X, y)

    y_pred, sigma, _ = gp.predict(x, return_std=True)

    visualize(X, x.squeeze(), y, y_pred.squeeze(), sigma, dy)