import torch 
from torch import nn
from torch.nn import functional as F
from torch.optim import SGD
from utils import get_two_moons

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt


class SVM:
    def __init__(self, kernel, gamma, C, eps=1e-2):
        self.kernel = kernel
        self.gamma = gamma
        self.C = C
        self.eps = eps

    def fit(self, X, y):
        self.X = X
        self.y = y[:, None]

        N, D = X.shape
        y = (y * 2 - 1)[..., None]

        # step 1: Apply kernel method
        pdist = (X[:, None, :] - X[None, :, :]).norm(2, -1)**2
        K = torch.exp(-self.gamma * pdist)

        # step 2: train network
        net = nn.Linear(N, 1)

        optimizer = SGD(net.parameters(), lr=1e-2, weight_decay=self.C)

        prev_loss = 1e16

        with torch.enable_grad():
            while True:

                loss = F.relu(1 - y * net(K)).pow(2).mean()

                # print(loss.item())

                if abs(loss.item() - prev_loss) < 1e-4 or torch.isnan(loss) or torch.isinf(loss):
                    break
                else:
                    prev_loss = loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # step 3: get support vectors
        predict = net(K)
        mask = (abs(predict - y) < self.eps) # & (y * predict > 0)
        indices = mask.squeeze().nonzero()[:, 0]

        N = len(indices)
        print('num support vectors:', N)

        self.X = X[indices]

        # step 4: retrain net with support vectors
        pdist = (X[:, None, :] - self.X[None, :, :]).norm(2, -1)**2
        K = torch.exp(-self.gamma * pdist)

        net = nn.Linear(N, 1)

        optimizer = SGD(net.parameters(), lr=1e-2, weight_decay=self.C)

        prev_loss = 1e16

        with torch.enable_grad():
            while True:

                loss = F.relu(1 - y * net(K)).pow(2).mean()

                # print(loss.item())

                if abs(loss.item() - prev_loss) < 1e-4 or torch.isnan(loss) or torch.isinf(loss):
                    break
                else:
                    prev_loss = loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.X = X[indices]

        self.net = net

    def predict(self, X):
        pdist = (X[:, None, :] - self.X[None, :, :]).norm(2, -1)**2
        K = torch.exp(-self.gamma * pdist)

        y = self.net(K).sign().squeeze()

        y = (y + 1) * 0.5

        return y 


def visualize(inputs, targets, text='', colors=None):
    classes = targets.unique()

    if colors is None:
        colors = ['r', 'b']

    _, ax = plt.subplots()

    for k, c in zip(classes, colors):
        input = inputs[targets==k]
        ax.scatter(input[:, 0], input[:, 1], c=c, label='Class {}'.format(k.item()))
    
    ax.legend()
    plt.savefig('figures/chapter8_{}.png'.format(text))


if __name__ == '__main__':
    train_inputs, test_inputs, train_targets, test_targets = get_two_moons(1500, 0.3, 0.2)

    model = SVM(kernel='rbf', gamma=10, C=1e-1, eps=1e-2)

    with torch.no_grad():
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

    