# tt
# 2021.7.28
# Recommender system with SVD

import numpy as np
import numpy.linalg as la
import pandas as pd

def load_data():
    # Load csv data
    header = ["u_id", "i_id", "rate", "ts"]
    data = pd.read_csv("./u.data", sep='\t', names=header)
    # Build ndarray
    m, n = data.u_id.max(), data.i_id.max()
    res = np.zeros((m, n))
    for line in data.itertuples():
        res[line[1] - 1][line[2] - 1] = line[3]
    return res

def eclud_sim(X, Y):
    dist = la.norm(X - Y)
    return 1.0 / (1.0 + dist)  # normalize to [0, 1]

def pears_sim(X, Y):
    if len(X) < 3:
        return 1.0
    r = np.corrcoef(X, Y, rowvar=1)[0][1]
    return 0.5 + 0.5 * r  # normalize to [0, 1]

def cos_sim(X, Y):
    XY = float(X.dot(Y))
    XYnorm = la.norm(X) * la.norm(Y)
    cosa = XY / XYnorm
    return 0.5 + 0.5 * cosa  # normalize to [0, 1]

if __name__ == "__main__":
    data = load_data()

