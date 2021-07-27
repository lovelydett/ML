import numpy as np

def self_def_dtype():
    # dtype can be defined like a struct
    student = np.dtype([('id', np.int32), ('age', np.int64), ('score', np.float64)])
    a = np.array([[1, 23, 99], [3, 19, 83.5]], dtype=student)
    print(f'a.dtype:\n{a.dtype}')
    print(f'a:\n{a}')


if __name__ == "__main__":
    self_def_dtype()
