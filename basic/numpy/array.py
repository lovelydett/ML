# ndarray (n-dimensional array)
import numpy as np

def check_ndarray():
    # ndarray is essentially an index structure (see c)
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([[1, 2.0, 3], [4, 5, 6]])  # dtype would be float64
    c = np.array([[1, 2, [10, 11]], [4, 5, 6]], dtype=object)  # dtype would be object
    print(c.dtype)
    print(c.shape)

def check_matrix():
    m = np.matrix([[1, 2, 3], [4, 5, 6]]) # matrix must be 2-dimensional!
    print(m.transpose()) # this doesnt modify itself

def zeros_ones():
    a = np.ones((2, 3), dtype=np.float64, order='F') # get a 2*3 array with ones
    b = np.zeros((2, 3, 4, 5), dtype='float64', order='F')  # or zeros
    print(a.shape)
    print(b.shape)

def ndarray_slice():
    l = [[[i * j * k for k in range(1, 5)] for j in range(1, 4)] for i in range(1, 3)]
    a = np.array(l, dtype='int64', order='A')
    print(f'a.shape: {a.shape}')
    print(f'a[1]:\n{a[1]}')
    print(f'a[1, 2]:\n{a[1, 2]}')
    print(f'a[:, 2]:\n{a[:, 2]}')

    b = a[:, 2] # modify b also modify a[:, 2]
    b *= -1
    print(f'b:\n{b}')
    print(f'a:\n{a}')

    print(f'a[1, :2, 1:3]:\n{a[1, :2, 1:3]}')
    a[1, :2, 1:3] = np.zeros(a[1, :2, 1:3].shape) # multi-dimensional slice can be operated as a whole
    print(f'a:\n{a}')

def ndarray_axis():
    l = [[[i * j * k for k in range(1, 5)] for j in range(1, 4)] for i in range(1, 3)]
    a = np.array(l, dtype='int64', order='A')
    print(f'a.shape:\n{a.shape}')
    print(f'repeat 3 by axis=0:\n{a.repeat(3, axis=0).shape}') # repeat those 2*3*4 by 3 times => 6*3*4
    print(f'repeat 3 by axis=1:\n{a.repeat(3, axis=1).shape}') # repeat those 3*4 by 3 times => 2*9*4
    print(f'repeat 3 by axis=2:\n{a.repeat(3, axis=-1).shape}') # repeat those 4 by 3 times => 2*3*12

def ndarray_complex():
    # numpy dtype supports complex
    a = np.array([2 + 3j, 3, 4], dtype='complex')
    print(a)

def ndarray_attributes():
    l = [[[i * j * k for k in range(1, 5)] for j in range(1, 4)] for i in range(1, 3)]
    a = np.array(l, dtype='int64', order='A')
    print(f'a.size:\n{a.size}')
    print(f'a.shape:\n{a.shape}')
    print(f'a.ndim:\n{a.ndim}') # namely rank
    print(f'a.itemsize:\n{a.itemsize}')

if __name__ == "__main__":
    ndarray_attributes()

