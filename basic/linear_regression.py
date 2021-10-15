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

def forward(params, input_x):
    result = params[0]
    for i in range(input_x.shape[0]):
        result += params[i + 1] * input_x[i]
    return result

def backward(params, input_x):
    pass

def fit_gradient_descent(x, y):
    num_features = x.shape[0]
    params = np.random.normal(0, 1, size=(num_features + 1, )) # Randomly initialize params





if __name__ == "__main__":
    x, y, params = generate_data(size=10, num_features=1)



