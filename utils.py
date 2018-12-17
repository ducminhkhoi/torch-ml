import numpy as np
np.random.seed(1991)

import torch 
from sklearn.datasets import load_iris, make_moons, make_swiss_roll, make_blobs, make_spd_matrix
from sklearn.model_selection import train_test_split

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_iris(val_ratio=0, normalized=False):
    inputs, targets = load_iris(True)

    indices = np.arange(inputs.shape[0])
    np.random.shuffle(indices)

    num_train = int(len(indices) * (1-val_ratio))

    train_inputs = inputs[indices[:num_train]]
    train_targets = targets[indices[:num_train]]

    if val_ratio > 0:
        test_inputs = inputs[indices[num_train:]]
        test_targets = targets[indices[num_train:]]

    if normalized:
        mean = train_inputs.mean(0)
        train_inputs -= mean
        std = train_inputs.std(0)
        train_inputs /= std

        if val_ratio > 0:
            test_inputs -= mean
            test_inputs /= std

    train_inputs = torch.from_numpy(train_inputs).float()
    train_targets = torch.from_numpy(train_targets)

    if val_ratio > 0:
        test_inputs = torch.from_numpy(test_inputs).float()
        test_targets = torch.from_numpy(test_targets)

    train_inputs = train_inputs.to(device)
    train_targets = train_targets.to(device)

    if val_ratio > 0:
        test_inputs = test_inputs.to(device)
        test_targets = test_targets.to(device)

        return train_inputs, train_targets, test_inputs, test_targets
    else:
        return train_inputs, train_targets


def get_two_moons(n_samples=100, noise=None, val_ratio=0):
    inputs, targets = make_moons(n_samples=n_samples, random_state=123, noise=noise)

    if val_ratio > 0:
        parts = train_test_split(inputs, targets, test_size=val_ratio)

        return [torch.from_numpy(x).to(device).float() for x in parts]
    else:
        return [torch.from_numpy(x).to(device).float() for x in [inputs, targets]]


def get_swiss_roll():
    X, y = make_swiss_roll(n_samples=1500, random_state=123)

    inputs = torch.from_numpy(X).to(device).float()
    targets = torch.from_numpy(y).to(device)

    return inputs, targets


def get_blobs():
    X, y = make_blobs(1500, random_state=170)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]

    X = X @ transformation

    inputs = torch.from_numpy(X).to(device).float()
    targets = torch.from_numpy(y).to(device)

    return inputs, targets


def get_spd(n_matrices, n_dim):
    return torch.stack([torch.from_numpy(make_spd_matrix(n_dim)) for _ in range(n_matrices)]).float()

def cov(m):
    x = m - m.mean(1, keepdim=True)
    cov = 1 / x.size(1) * (x @ x.t())
    return cov


def kron(a, b):
    return a.view(-1, 1) @ b.view(1, -1)

if __name__ == '__main__':
    get_iris()