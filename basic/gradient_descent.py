import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# function
def Fun(x, y):
    return x - y + 2 * x * x + 2 * x * y + y * y

# Partial derivative of x
def PxFun(x, y):
    return 1 + 4 * x + 2 * y

# Partial derivative of y
def PyFun(x, y):
    return -1 + 2 * x + 2 * y

if __name__ == "__main__":
    fig = plt.figure() # figure object
    ax = Axes3D(fig) # make 3D object on fig
    X, Y = np.mgrid[-2:2:40j, -2:2:40j] # here 40j means divide into 40 intervals
    Z = Fun(X, Y) # get Z coordinates
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap="rainbow")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # plt.show() # draw the original function

    # now do gradient descent
    step = 0.0008 # step length
    x, y = 0, 0
    trail_x, trail_y, trail_z = [], [], []
    nx, ny = x, y
    while True:
        trail_x.append(x)
        trail_y.append(y)
        trail_z.append(Fun(x, y))
        nx -= step * PxFun(x, y)
        ny -= step * PyFun(x, y)
        if Fun(x, y) -  Fun(nx, ny) < 7e-9:
            break
        x, y = nx, ny
    ax.plot(trail_x, trail_y, trail_z, 'r')
    plt.show()
