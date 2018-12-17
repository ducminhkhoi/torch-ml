import torch 
from torch import nn
from torch.nn import functional as F
from torch.optim import SGD
from utils import get_two_moons
from copy import deepcopy
from torch.autograd import grad

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class GradientBoost:
    def __init__(self, learner, loss_fn, M, lr):
        self.learner = learner
        self.loss_fn = loss_fn
        self.M = M
        self.lr = lr

    def train(self, variables, loss_fn, func, X, y, lr=1e-1):
        optimizer = SGD(variables, lr=lr)
        prev_loss = 1e16

        # print('------------------------>')

        while True:
            loss = loss_fn(func(*X), y)

            # print(loss.item())

            if abs(loss.item() - prev_loss) < 1e-4 or torch.isnan(loss) or torch.isinf(loss):
                break
            else:
                prev_loss = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return func(*X)

    def fit(self, X, y):
        N, D = X.shape
        y = y[:, None]
        # 1. Initialize model with a constant value:
        t = deepcopy(self.learner)
        F = self.train(t.parameters(), self.loss_fn, lambda X: t(X), [X], y)

        self.first_learner = t
        self.hs = []

        # 2.
        for m in range(self.M):

            # 1. Compute so-called pseudo-residuals:
            r = -grad(self.loss_fn(F, y), F)[0]
            print(r.sum().item())

            # 2. Fit a base learner (e.g. tree) {\displaystyle h_{m}(x)} {\displaystyle h_{m}(x)} to pseudo-residuals,
            t = deepcopy(self.learner)
            h = self.train(t.parameters(), nn.MSELoss(), lambda X: t(X), [X], r)
            self.hs.append(t)

            # 4. Update the model
            F = F + self.lr * h

    def predict(self, X):
        with torch.no_grad():
            y = self.first_learner(X)
            for h in self.hs:
                y += self.lr * h(X)

            result = torch.where(y > 0, torch.ones_like(y), torch.zeros_like(y))
            return result.squeeze()


def visualize(inputs, targets, text='', colors=None):
    classes = targets.unique()

    if colors is None:
        colors = ['r', 'b']

    _, ax = plt.subplots()

    for k, c in zip(classes, colors):
        input = inputs[targets==k]
        ax.scatter(input[:, 0], input[:, 1], c=c, label='Class {}'.format(k.item()))
    
    ax.legend()
    plt.savefig('figures/chapter9_{}.png'.format(text))


if __name__ == '__main__':
    train_inputs, test_inputs, train_targets, test_targets = get_two_moons(1500, 0.3, 0.2)

    learner = MLP(2, 2, 1)
    loss_fn = nn.BCEWithLogitsLoss()

    model = GradientBoost(learner, loss_fn, M=20, lr=1e-1)

    model.fit(train_inputs, train_targets)

    train_predict = model.predict(train_inputs)

    acc = (train_predict == train_targets).sum().item() / len(train_inputs)

    print(acc)

    visualize(train_inputs, train_targets, text='train_gt')
    visualize(train_inputs, train_predict, text='train_svm1')

    test_predict = model.predict(test_inputs)

    acc = (test_predict == test_targets).sum().item() / len(test_inputs)

    print(acc)

    visualize(test_inputs, test_targets, text='test_gt')
    visualize(test_inputs, test_predict, text='test_svm1')

    