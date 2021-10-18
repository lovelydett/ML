# tt
# 2021.7.27
# Singular value decomposition

import numpy as np
from numpy import linalg as la

def ATandA():
    A = np.array([[1, 5, 7, 6, 1], [2, 1, 10, 4, 4], [3, 6, 7, 5, 2]])
    AAT = A.dot(A.T) # dot is matrix multiply but '*' is vector multiply
    ATA = A.T.dot(A)
    eval1, evec1 = la.eig(AAT)
    eval2, evec2 = la.eig(ATA)
    print(f'rank of A:{la.matrix_rank(A)}')
    print(f'rank of AAT:{la.matrix_rank(AAT)}')
    print(f'rank of ATA:{la.matrix_rank(ATA)}')
    print(f'eigen values of AAT:{np.round(eval1)}')
    print(f'eigen values of ATA:{np.round(eval2)}')


def SVD():
    A = np.array([[1, 5, 7, 6, 1], [2, 1, 10, 4, 4], [3, 6, 7, 5, 2]])
    U, s, VT = la.svd(A) # s is all eigen values

    # make eigen values to be sigma matrix
    sigma = np.zeros(A.shape)
    sigma[:len(s), :len(s)] = np.diag(s)

    # rebuild A with U, sigma and VT
    A_ = U.dot(sigma.dot(VT))
    print(f"A:\n{A}")
    print(f"A_:\n{A_}")
    print(f"np.allclose(A, A_):{np.allclose(A, A_)}")


def approximate():
    A = np.array([[1, 5, 7, 6, 1], [2, 1, 10, 4, 4], [3, 6, 7, 5, 2]])
    print(f"A:\n{A}")
    U0, s, VT0 = la.svd(A)
    sigma0 = np.zeros(A.shape)
    sigma0[:len(s), :len(s)] = np.diag(s)

    # we use only the top k eigen values to approximate original A
    for k in [1, 2, 3]:
        U = U0[:, :k]
        sigma = sigma0[:k, :k]
        VT = VT0[:k, :]
        A_ = U.dot(sigma.dot(VT))
        print(f"k = {k}")
        print(f"A_:\n{np.round(A_, 2)}")


if __name__ == "__main__":
    approximate()
