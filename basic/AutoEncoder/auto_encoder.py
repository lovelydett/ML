'''
tt
2021-10-25
Auto-Encoder pytorch implementation
'''

import torch
import numpy as np

class AutoEncoder(torch.nn.Module):
    def __init__(self, data_len=128, encode_len=32):
        super(AutoEncoder, self).__init__()
        self.dense1 = torch.nn.Linear(in_features=data_len, out_features=encode_len)
        self.dense2 = torch.nn.Linear(in_features=encode_len, out_features=data_len)

        self.dense1.weight.data.normal_(0, 0.1)
        self.dense2.weight.data.normal_(0, 0.1)

        self.dense1 = self.dense1.cuda()
        self.dense2 = self.dense2.cuda()

    def forward(self, input):
        input = input.cuda()
        x = self.dense1(input)
        x = self.dense2(x)
        return x

    def encode(self, input):
        input = input.cuda()
        return self.dense1(input)

    def decode(self, input):
        input = input.cuda()
        return self.dense2(input)

def generate_data(size, data_len=128):
    data = np.zeros((size, data_len))
    for i in range(size):
        data[i][0] = np.random.randint(low=1, high=200)
        for j in range(1, data_len):
            if j % 2 == 1:
                data[i][j] = data[i][j - 1] + 3
            else:
                data[i][j] = data[i][j - 1] * 1.1
    return data

def train(X, epochs=100, batch_size=30):
    ae = AutoEncoder()

    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(ae.parameters())

    X = X.cuda()
    for i in range(epochs):
        batch_idx = np.random.randint(low=0, high=X.shape[0] - batch_size)
        batch = X[batch_idx:batch_idx + batch_size]
        output = ae.forward(batch)
        loss = loss_func(batch, output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch{i + 1}: loss = {loss}")

    return ae


def test(ae, X):
    loss_func = torch.nn.MSELoss()
    output = ae.forward(X)
    X = X.cuda()
    loss = loss_func(X.cuda(), output)
    print(f"Before:\n{X}")
    print(f"After:\n{output}")
    print(f"Diff:\n{abs(X - output) / X}")
    print(f"Test loss = {loss}")


def demo():
    data = torch.Tensor(generate_data(size=1000))
    X, X_test = data[:800], data[800:]
    ae = train(X, epochs=1000)
    test(ae, X_test)


if __name__ == "__main__":
    demo()


