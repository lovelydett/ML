# tt
# 2021.7.27
# Image suppression with SVD
import numpy
from PIL import Image
import numpy as np

def get_approximate_SVD1(data:numpy.ndarray, p=100):
    '''
    Approximate matrix by SVD, with top p% of SUM of total singular values
    :param data: input matrix
    :param p: top p% of SUM of singular values
    :return: approximated
    '''
    U, s, VT = np.linalg.svd(data) # do SVD
    sigma = np.zeros(data.shape)
    sigma[:len(s), :len(s)] = np.diag(s)
    total = np.sum(s) * p / 100
    count, k = 0, -1
    while count < total:
        count += s[k]
        k += 1
    res = U[:, :k].dot(sigma[:k, :k].dot(VT[:k, :]))
    res[res < 0] = 0
    res[res > 255] = 255
    return np.rint(res).astype('uint') # Round elements of the array to the nearest integer.

def get_approximate_SVD2(data:numpy.ndarray, p=100):
    '''
    Approximate matrix by SVD, with top p% of NUMBER of total singular values
    :param data: input matrix
    :param p: top p% of NUMBER of singular values
    :return: approximated
    '''
    U, s, VT = np.linalg.svd(data) # do SVD
    sigma = np.zeros(data.shape)
    sigma[:len(s), :len(s)] = np.diag(s)
    k = int(len(s) * p / 100)
    print(k, len(s))
    res = U[:, :k].dot(sigma[:k, :k].dot(VT[:k, :]))
    res[res < 0] = 0
    res[res > 255] = 255
    return np.rint(res).astype('uint') # Round elements of the array to the nearest integer.

def suppress_image(filepath):
    if not filepath.endswith((".jpg", ".png")):
        return
    img = np.array(Image.open(filepath, 'r').convert('L'))
    for p in range(100, 40, -10):
        img_ = get_approximate_SVD1(img, p)
        # img_ = get_approximate_SVD2(img, p) # even p = 50 for SVD2 we have a very accurate approximation
        Image.fromarray(img_.astype('float64')).show(title=str(p))


if __name__ == "__main__":
    suppress_image("./1.jpg")