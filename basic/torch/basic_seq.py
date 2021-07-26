import torch
import torch.nn as nn
import numpy as np

in_size, hidden_size, out_size, batch_size = 20, 10, 1, 10

# Mock data
train_size, test_size = 200, 50
params = np.random.random((in_size, )) # this is what to learn
X = torch.Tensor(np.random.random((train_size + test_size, in_size)))
Y = torch.Tensor(np.zeros((train_size + test_size, 1), dtype='float64'))
for i in range(train_size + test_size):
    for p in range(in_size):
        Y[i][0] += X[i][p] * params[p]

# Build sequential model
model = nn.Sequential(
    nn.Linear(in_size, hidden_size), # Dense layer
    nn.ReLU(),
    nn.Linear(hidden_size, out_size), # Output layer
    nn.ReLU(),
)

# Loss func and optimizer
loss_function = nn.L1Loss() # L1Loss = MAE
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-2)

# Train epochs:
epoch = 30
for i in range(30):
    # Forward
    y_hat = model(X[:train_size])
    # Compute loss
    loss = loss_function(y_hat, Y[:train_size])
    print('epoch: ', i,' loss: ', loss.item())
    # Init grad to be zero
    optimizer.zero_grad()
    # Backward
    loss.backward()
    # Update weights
    optimizer.step()

# Test:
y_hat = model(X[train_size:])
loss = loss_function(y_hat, Y[train_size:])
print("Test loss: ", loss.item())

