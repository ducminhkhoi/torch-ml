import torch 
from torch import nn
from torch.nn import functional as F
from torch.optim import SGD
from utils import get_iris, device, cov, get_two_moons, get_swiss_roll, kron
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA2
from sklearn.decomposition import PCA as PCA2
from sklearn.decomposition import KernelPCA as KernelPCA2
from sklearn.manifold import LocallyLinearEmbedding as LLE2

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt


class LDA:
    def __init__(self, n_dims):
        self.n_dims = n_dims

    def fit_transform(self, X, y):
        N, d = X.shape
        classes = y.unique()

        # compute S_w
        S_w = torch.zeros(d, d)

        for c in classes:
            x_c = X[y==c]
            pi_c = x_c.shape[0]/N
            S_w += pi_c * cov(x_c.t())

        # compute S_b
        C = cov(X.t())
        S_b = C - S_w

        M = S_w.inverse() @ S_b

        # compute eigen value and eigen vector
        eigvals, eigvecs = M.eig(True)

        indices = eigvals[:, 0].sort(descending=True)[1][:self.n_dims]
        
        self.w = eigvecs[:, indices]

        return X @ self.w 


class PCA:
    def __init__(self, n_dims):
        self.n_dims = n_dims

    def fit_transform(self, X):
        self.mean =  X.mean(0, keepdim=True)

        X -= self.mean

        C = cov(X.t())

        eigvals, eigvecs = C.eig(True)

        indices = eigvals[:, 0].sort(descending=True)[1][:self.n_dims]
        
        self.w = eigvecs[:, indices]

        return X @ self.w


class AE:
    def __init__(self, n_dims):
        self.n_dims = n_dims

    def fit_transform(self, X):
        input_dim = X.shape[1]
        encoder = nn.Linear(input_dim, self.n_dims)
        decoder = nn.Linear(self.n_dims, input_dim)

        net = nn.Sequential(encoder, decoder)

        optimizer = SGD(net.parameters(), lr=1e-2)
        prev_loss = 1e16

        with torch.enable_grad():
            while True:
                x_ = encoder(X)
                x = decoder(x_)

                loss = F.mse_loss(x, X)

                print(loss.item())
                
                if abs(loss.item() - prev_loss) < 1e-4 or torch.isnan(loss):
                    break
                else:
                    prev_loss = loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        y = encoder(X)

        return y


class KernelPCA:
    def __init__(self, n_dims, gamma):
        self.n_dims = n_dims
        self.gamma = gamma

    def fit_transform(self, X):
        # compute the euclidean distance and kernel matrix
        pdist = (X[:, None, :] - X[None, :, :]).norm(2, -1)**2
        K = torch.exp(-self.gamma * pdist)

        # centering the Kernel matrix
        N = K.shape[0]
        one_n = torch.ones(N, N) / N
        K -= one_n @ K + K @ one_n - one_n @ K @ one_n

        # compute eigenvalues, eigenvectors of K
        eigvals, eigvecs = K.eig(True)

        indices = eigvals[:, 0].sort(descending=True)[1][:self.n_dims]
        
        return eigvecs[:, indices]

    
class LLE:
    def __init__(self, n_dims, k):
        self.n_dims = n_dims
        self.k = k

    def fit_transform(self, X):
        N = X.shape[0]
        # perform KNN
        N = X.shape[0]
        pdist = (X[:, None, :] - X[None, :, :]).norm(2, -1)

        knn = pdist.topk(self.k, 1, largest=False)[1]

        # compute weigths matrix W
        W = torch.zeros(N, self.k)

        for i in range(N):
            Z = X[knn[i]] - kron(torch.ones(self.k, 1), X[i])
            C = Z @ Z.t()

            C += torch.eye(self.k) * 1e-3 * C.trace()

            w = torch.gesv(torch.ones(self.k, 1), C)[0].t()

            W[i] = w / w.sum()

        # compute Y
        M = torch.eye(N)
        for i in range(N):
            w = W[i:i+1].t()
            ww = w @ w.t()

            for k in range(self.k):
                M[i, knn[i, k]] -= w[k].item()
                M[knn[i, k], i] -= w[k].item()
                for l in range(self.k):
                    M[knn[i, k], knn[i, l]] += ww[k, l]

        eigvals, eigvecs = M.eig(True)

        indices = eigvals[:, 0].sort()[1][1:self.n_dims+1] 

        y = eigvecs[:, indices]

        return y

    def fit_transform2(self, X):
        # perform KNN
        N, D = X.shape
        pdist = (X[:, None, :] - X[None, :, :]).norm(2, -1)

        knn = pdist.topk(self.k+1, 1, largest=False)[1][:, 1:].contiguous()

        mask = torch.zeros_like(pdist)
        indices = torch.arange(N).repeat(self.k, 1).t().contiguous().view(-1)
        mask[indices, knn.view(-1)] = 1.

        with torch.enable_grad():
            # optimize to find W
            linear = nn.Linear(D, D*2)

            optimizer = SGD(linear.parameters(), lr=1e-2)
            prev_loss = 1e16

            while True:
                x = linear(X)
                W = x @ x.t()
                W = torch.where(mask > 0, W, torch.full_like(W, -1e16))
                W = F.softmax(W, 1)

                loss = F.mse_loss(W @ X, X)

                print(loss.item())

                if abs(loss.item() - prev_loss) < 1e-4:
                    break
                else:
                    prev_loss = loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            W = W.detach()

            # optimize to find Y
            Y = nn.Parameter(torch.randn(N, self.n_dims)).to(device)
            optimizer = SGD([Y], lr=1e-2)
            prev_loss = 1e16

            while True:
                loss = F.mse_loss(W @ Y, Y)

                if abs(loss.item() - prev_loss) < 1e-4:
                    break
                else:
                    prev_loss = loss.item()

                print(loss.item())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return Y.data


def visualize(inputs, targets, text='train_original'):
    classes = targets.unique()
    colors = ['r', 'g', 'b']

    _, ax = plt.subplots()

    for k, c in zip(classes, colors):
        input = inputs[targets==k]
        ax.scatter(input[:, 0], input[:, 1], c=c, label='Class {}'.format(k.item()))
    
    ax.legend()
    plt.savefig('figures/chapter6_{}.png'.format(text))


def visualize3D(X, y, X_r, text):
    from mpl_toolkits.mplot3d import Axes3D
    Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(211, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.Spectral)
    ax.set_title("Original data")

    ax = fig.add_subplot(212)
    ax.scatter(X_r[:, 0], X_r[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.axis('tight')
    # plt.xticks([]), plt.yticks([])
    plt.title('Projected data')
    plt.savefig('figures/chapter6_{}.png'.format(text))


if __name__ == '__main__':
    
    model_name = 'LLE'
    dataset_name = 'swiss_roll'

    if model_name == 'LDA':
        model = LDA(2)
    elif model_name == 'PCA':
        model = PCA(2)
    elif model_name == 'KernelPCA':
        model = KernelPCA(2, 15)
    elif model_name == 'LLE':
        # model = LLE2(12, 2)
        model = LLE(2, 12)
    elif model_name == 'AE':
        model = AE(2)

    train_inputs, train_targets = globals()['get_'+dataset_name]()

    with torch.no_grad():
        if model_name == 'LDA':
            new_train_inputs = model.fit_transform(train_inputs, train_targets)
        else:
            new_train_inputs = model.fit_transform(train_inputs)

    visualize3D(train_inputs, train_targets, new_train_inputs, text='{}_{}'.format(dataset_name, model_name))

    # visualize(train_inputs, train_targets, text='train_original')

    # visualize(train_inputs, train_targets, text='{}_train_original'.format(dataset_name))
    # visualize(new_train_inputs, train_targets, text='{}_train_{}'.format(dataset_name, model_name))

    # visualize(train_inputs, train_targets, text='two_moons_train_original')
    # visualize(new_train_inputs, train_targets, text='two_moons_train_{}'.format(model_name))
