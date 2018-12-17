import torch 
from torch import nn
from torch.nn import functional as F
from torch.optim import SGD
from sklearn import cluster, mixture
import numpy as np
from utils import get_iris, device


class RBFNet(nn.Module):
    def __init__(self, centers, num_out):
        super().__init__()

        self.num_out = num_out
        self.num_centers = centers.shape[0]

        self.centers = nn.Parameter(torch.from_numpy(centers).float())
        self.sigma = nn.Parameter(torch.ones(1,self.num_centers)/10)

        self.linear = nn.Linear(self.num_centers, num_out)

    def forward(self, x):
        # 1st layer
        x = torch.exp(-((x[:, None, :] - self.centers[None, :, :])**2).sum(2)/(2*self.sigma**2))

        # 2nd layer
        x = self.linear(x)

        return x


class RBFNormalizedNet(nn.Module):
    def __init__(self, centers, num_out):
        super().__init__()

        self.num_out = num_out
        self.num_centers = centers.shape[0]

        self.centers = nn.Parameter(torch.from_numpy(centers).float())
        self.sigma = nn.Parameter(torch.ones(1,self.num_centers)/10)

        self.linear = nn.Linear(self.num_centers, num_out)

    def forward(self, x):
        # 1st layer
        x = F.softmax(-((x[:, None, :] - self.centers[None, :, :])**2).sum(2)/(2*self.sigma**2), 1)

        # 2nd layer
        x = self.linear(x)

        return x


class RBFNet2(nn.Module):
    def __init__(self, centers, sigmas, num_out):
        super().__init__()

        self.num_out = num_out
        self.num_centers = centers.shape[0]

        self.centers = nn.Parameter(torch.from_numpy(centers).float())
        self.sigma = nn.Parameter(torch.from_numpy(sigmas).float())

        self.linear = nn.Linear(self.num_centers, num_out)

    def forward(self, x):
        # 1st layer
        x = torch.exp((-((x[:, None, :] - self.centers[None, :, :])**2)/(2*self.sigma[None, :, :]**2)).sum(2))

        # 2nd layer
        x = self.linear(x)

        return x

def train():
    prev_loss = 1e16

    rbf.train()
    for epoch in range(num_epochs):

        outputs = rbf(train_inputs)
        loss = criterion(outputs, train_targets)

        if (epoch+1) % 1000 == 0:
            print(loss.item())

        if abs(prev_loss - loss.item()) < 1e-4:
            break
        else:
            prev_loss = loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test():
    rbf.eval()
    outputs = rbf(test_inputs)
    pred = outputs.argmax(1)
    acc = (pred == test_targets).sum().float()/test_targets.shape[0]

    print(acc.item())


if __name__ == '__main__':
    train_inputs, train_targets, test_inputs, test_targets = get_iris()

    # get centers
    kmeans = cluster.KMeans(5)
    kmeans.fit(train_inputs)
    centers = kmeans.cluster_centers_
    rbf = RBFNet(centers, 3)

    # # get centers and sigmas
    # gmm = mixture.GaussianMixture(5, covariance_type='diag')
    # gmm.fit(train_inputs)
    # centers = gmm.means_
    # sigmas = gmm.covariances_
    # rbf = RBFNet2(centers, sigmas, 3)

    optimizer = SGD(rbf.parameters(), lr=1e-1)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 1000000

    train()
    test()




