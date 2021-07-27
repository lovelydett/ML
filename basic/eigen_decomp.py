# Matrix eigen decomposition
# A = Q * sigma * Q^-1

import numpy as np

A = np.array([[4, 2], [1, 5]])
print(f"A:\n{A}")

# get eigen values and eigen vectors
eigen_val, Q = np.linalg.eig(A)
print(f"eigen values:\n{eigen_val}")
print(f"Q:\n{Q}")

# get a diagonal matrix of eigenvalues
sigma = np.diag(eigen_val)
print(f"sigma:\n{sigma}")

# validate the columns of Q are eigen vectors
print(Q[:, 0].dot(Q[:, 0]))
print(Q[:, 1].dot(Q[:, 1]))

# compute the decomposition
A_ = Q.dot(sigma.dot(np.linalg.inv(Q)))
print(f"A_:\n{A_}")
print(f"np.allclose(A, A_):{np.allclose(A, A_)}")
