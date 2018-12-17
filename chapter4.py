import torch
from torch import nn
from torch.optim import SGD
import numpy as np
from utils import get_iris, device

# XOR and AND examples
anddata = torch.tensor([[0,0,0],[0,1,0],[1,0,0],[1,1,1]]) 
xordata = torch.tensor([[0,0,0],[0,1,1],[1,0,1],[1,1,0]])

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = x.sigmoid()
        x = self.fc2(x)
        return x


def training(data, num_epochs=2000):
    net = MLP(2, 3, 1)
    optimizer = SGD(net.parameters(), lr=1e-2, momentum=0.1)
    criterion = nn.MSELoss()

    # training:
    for t in range(num_epochs):
        for x in data:
            x = x[None, :].float()
            y = net(x[:, :2])

            loss = criterion(y, x[:, -1])

            if (t+1) % 1000 == 0:
                print(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # recall:
    y = net(data[:, :2].float())
    print(torch.where(y > 0.5, torch.ones_like(y), torch.zeros_like(y)), data[:, -1])
    

# training(anddata)
# training(xordata, 10000)

def train():
    prev_loss = 1e16
    for epoch in range(num_epochs):
        outputs = mlp(train_inputs)
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
    outputs = mlp(test_inputs)
    pred = outputs.argmax(1)
    acc = (pred == test_targets).sum().float()/test_targets.shape[0]

    print(acc.item())


if __name__ == '__main__':
    train_inputs, train_targets, test_inputs, test_targets = get_iris()

    mlp = MLP(4, 10, 3)
    optimizer = SGD(mlp.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 100000
    train()
    test()



