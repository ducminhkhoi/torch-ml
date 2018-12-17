import torch

# Perceptron, OR example

inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = torch.tensor([[0],[1],[1],[1]])

# add 1 to represent bias:
inputs = torch.cat([torch.ones(inputs.shape[0], 1), inputs.float()], 1)[:, None]
targets = targets.float()

n_in = 2
n_out = 1
eta = 0.25

# initialization
weight = torch.rand(n_in + 1, n_out) * 0.1 - 0.05

# training
for t in range(10):
    for input, target in zip(inputs, targets):
        a = input @ weight
        y = 1 if a > 0 else 0

        weight -= (eta * (y - target) * input).t()

inputs = inputs.squeeze()
# recall (evaluation)
a = inputs @ weight
y = torch.where(a > 0, torch.ones_like(a), torch.zeros_like(a))

print(y)
