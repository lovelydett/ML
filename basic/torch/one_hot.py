# tt
# 2021.9.29
# One-hot encoding with pytorch

import torch
import numpy as np


def one_hot():
    origin_labels = torch.tensor(np.random.randint(0, 15, size=(2, 3)))
    print(f"origin_labels:\n{origin_labels}")
    one_hot_codes = torch.nn.functional.one_hot(origin_labels, 15)
    print(f"one_hot_codes:\n{one_hot_codes}")


if __name__ == "__main__":
    one_hot()