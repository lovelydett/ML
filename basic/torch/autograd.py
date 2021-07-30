# tt
# 2021.7.30
# Pytorch autograd

import torch as t
from torch.autograd import Variable


def variable():
    x = Variable(t.ones(2, 2), requires_grad=True)  # create a variable from Tensor
    print(f"x:\n{x}")
    y = x.sum()
    print(f"y = x.sum():{y}")
    print(f"y.grad_fn:\n{y.grad_fn}")

    print(f"x.grad before bp:\n{x.grad}")
    for i in range(1, 5):
        y.backward() # backward propagation
        print(f"x.grad after {i} bp:\n{x.grad}") # x.grad will accumulate!!

    x.grad.data.zero_()
    print(f"x.grad.data.zero_():\n{x.grad}")
    y.backward()

    a = x.data # a and x shares memory
    a[0][0] = -1
    print(f"x:\n{x}")



if __name__ == "__main__":
    variable()
