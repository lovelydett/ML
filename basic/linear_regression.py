'''
tt
2021-10-14
Linear regression
'''

import numpy as np
import matplotlib.pyplot as plt

def generate_data(size=100, num_features=5):
    params = np.random.randint(1, 10, size=(num_features + 1, ))
    x = np.random.uniform(0, size, size=(size, num_features)) # +1 for Theta0
    y = np.random.normal(0, 6, size=(size, )) # This is initial noise
    for i in range(size):
        value = params[0]
        for j in range(1, num_features + 1):
            value += x[i][j - 1] * params[j]
        y[i] += value
    return x, y, params


# forward a batch
def forward(params, x):
    y_hat = np.zeros((x.shape[0], ))
    for i in range(x.shape[0]): # for each record in batch
        y_hat[i] = params[0]
        for j in range(x[i].shape[0]):
             y_hat[i] += params[j + 1] * x[i][j]
    return y_hat


# compute grad on a batch, return dJ/dTheta for each Theta
def get_grad(params, x, y, y_hat):
    batch_size = x.shape[0]
    grads = np.zeros(params.shape)
    for j in range(params.shape[0] - 1): # remember theta0 is not used here
        for i in range(x.shape[0]):
            grads[j + 1] += (y_hat[i] - y[i]) * x[i][j]
        grads[j + 1] /= batch_size

    # Handle theta0
    for i in range(x.shape[0]):
        grads[0] += (y_hat[i] - y[i]) * 1 # 1 for theta0
    grads[0] /= batch_size

    return grads


def compute_loss(y, y_hat):
    loss = y - y_hat
    loss **= 2
    loss = loss.sum()
    loss /= 2 * y.shape[0]
    return loss


def fit_gradient_descent(x, y, batch_size=20, epochs=10, lr=1e-2):
    batch_size, num_features = x.shape[0], x.shape[1]
    params = np.random.normal(0, 1, size=(num_features + 1, )) # Randomly initialize params
    loss = []
    batch_size = batch_size if batch_size <= x.shape[0] else x.shape[0]
    for i in range(epochs):
        i += 1
        # pick a batch
        b_idx = np.random.randint(0, x.shape[0] - batch_size + 1)
        x_batch = x[b_idx:b_idx + batch_size]
        y_batch = y[b_idx:b_idx + batch_size]
        y_hat = forward(params, x_batch)
        loss.append(compute_loss(y_batch, y_hat))
        grads = get_grad(params, x_batch, y_batch, y_hat)
        params -= lr * grads # Gradient descent!!!
        print(f"Epoch{i}/{epochs}, loss = {loss[-1]}")
    plt.plot(range(len(loss[1:])), loss[1:])
    plt.show()
    return params

if __name__ == "__main__":
    x, y, true_params = generate_data(size=100, num_features=20)
    params = fit_gradient_descent(x, y, lr=1e-5, epochs=1000)
    print(params)
    print(true_params)
    print(abs(params - true_params) / true_params)



