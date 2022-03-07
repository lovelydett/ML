# Train a simple xor network to get a .pt model file
# Author: Yuting Xie
# 2022.3.7

import torch
import torch.nn as nn
import numpy as np

def get_model(in_size, out_size, hidden_size):
    return nn.Sequential(
        nn.Linear(in_size, hidden_size), # Dense layer
        nn.ReLU(),
        nn.Linear(hidden_size, out_size), # Output layer
        nn.Sigmoid(),
    )

def generate_xor_data(size):
    x = [np.random.randint(0, 2, 2) for _ in range(size)]
    y = [[x[i][0] ^ x[i][1]] for i in range(size)]
    return x, y

if __name__ == "__main__":
    # Get train data
    X, Y = generate_xor_data(20)
    X, Y = np.array(X), np.array(Y)
    X, Y = torch.FloatTensor(X), torch.FloatTensor(Y)

    # Get the model
    net = get_model(2, 1, hidden_size=60)

    # Get loss function and optimizer
    cross_entropy = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr = 1e-2)
    optimizer.zero_grad()

    # Train
    epoch = 200
    for i in range(epoch):
        y_hat = net.forward(X)
        loss = cross_entropy(y_hat, Y)
        print(f'epoch{i + 1}: loss = {loss}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Test
    testX = torch.tensor([[0., 0.], [0., 1.], [1., 1.], [1., 0.]])
    testY = torch.tensor([[0.], [1.], [0.], [1.]])
    y_pred = net.forward(testX)
    print(f"Ground truth:\t{testY} \n Pred:\t{y_pred}")

    # Save model as .pt file for libtorch
    input_example = torch.tensor([[0., 0.]])
    traced_script_module = torch.jit.trace(net, input_example)
    traced_script_module.save("../model/xor_model.pt")