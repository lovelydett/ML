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

def check(x):
    return True

def func(l1, l2, l3):
    cnt, i, j, k = 0, 0, 0, 0
    res = []
    while cnt < 100:
        x = -1e5
        if i < len(l1):
            x = l1[i]
        if j < len(l2):
            x = max(x, l2[j])
        if k < len(l3):
            x = max(x, l3[k])
        if x == -1e5:
            break
        if check(x):
            res.append(x)
            cnt += 1
        if i < len(l1) and x == l1[i]:
            i += 1
        elif j < len(l2) and x == l2[j]:
            j += 1
        else:
            k += 1




if __name__ == "__main__":
    tensor()