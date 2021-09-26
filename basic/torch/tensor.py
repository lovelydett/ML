# tt
# 2021.7.30
# Pytorch tensor

import torch as t

def tensor():
    x = t.Tensor(5, 3) # only malloc, not inited
    print(f"Tensor(5, 3):\n{x}")

    y = t.rand(5, 3) # random distribution
    print(f"rand(5, 3):\n{y}")

    print(f"x + y:\n{x + y}")
    print(f"t.add(x, y):\n{t.add(x, y)}")

    res = t.Tensor(5, 3)
    t.add(x, y, out=res) # pre malloc for result Tensor
    print(f"pre-malloc res tensor:\n{res}")

    y.add_(x)
    print(f"y.add_(x):\n{y}") # xxx_() is inplace
    print(f"y.t():\n{y.t()}")

    a = t.ones(5) # get a 5-vec
    print(f"a:\n{a}")
    print(f"a.shape:\n{a.shape}")
    b = a.numpy() # b and a share the same memory!!!
    b[0] = -1
    print(f"now a:\n{a}")

    print(f"t.cuda.is_available():{t.cuda.is_available()}")
    x = x.cuda()
    y = y.cuda()
    print(f"x + y (in cuda):\n{x + y}")

    # squeeze
    a = t.tensor([[[1, 2, 3], [4, 5, 6]]])
    print(f"before squeeze_(0):\n{a.shape} ")
    a.squeeze_(0)
    print(f"after squeeze_(0):\n{a.shape} ")

if __name__ == "__main__":
    tensor()