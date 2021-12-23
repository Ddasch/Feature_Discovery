import cupy as cp
import numpy as np
import pandas as pd

from featurediscovery.kernels.monovariate.monovariate_kernels import *



def test_quadratic_kernel():
    x = cp.array([
        [1.0, -1.0],
        [2.0, -2.0],
        [3.0, -3.0],
        [4.0, -4.0],
        [5.0, -5.0],
        [6.0, -6.0],
    ])

    kernel = Quadratic_Kernel('raw')

    kernel.fit(x)

    x_quad = kernel.transform(x)

    x_exp = cp.array([
        [1.0, 1.0],
        [2.0*2, 2.0*2],
        [3.0*3, 3.0*3],
        [4.0*4, 4.0*4],
        [5.0*5, 5.0*5],
        [6.0*6, 6.0*6],
    ])

    np.testing.assert_array_equal(x_quad.get(), x_exp.get())



def test_gaussian_aprox():
    x = cp.array([
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0],
        [6.0],
    ])

    kernel = Gaussian_Kernel_Taylor_Aprox(degree=3, sigma=1)

    kernel.fit(x)

    x_quad = kernel.transform(x)

    print('')

    #np.testing.assert_array_equal(x_quad.get(), x_exp.get())



def test_sigmoid():
    x = cp.array([
        [1.0],
        [2.0],
        [3.0],
        [0.0],
        [0.5],
        [0.75],
        [-1.]
    ])

    kernel = Sigmoid_Kernel('raw')

    kernel.fit(x)

    x_quad = kernel.transform(x)

    print('')



def test_log_kernel():
    x = cp.array([
        [1.0, -1.0],
        [2.0, -2.0],
        [3.0, -3.0],
        [4.0, -4.0],
        [5.0, -5.0],
        [6.0, -6.0],
    ])

    kernel = Shifted_Log_Kernel('raw')

    kernel.fit(x)

    x_kernel = kernel.transform(x)

    epsilon = Shifted_Log_Kernel.epsilon
    x_exp = cp.array([
        [np.log(0.0 + epsilon), np.log(5.0+ epsilon)],
        [np.log(1.0 + epsilon), np.log(4.0+ epsilon)],
        [np.log(2.0 + epsilon), np.log(3.0+ epsilon)],
        [np.log(3.0 + epsilon), np.log(2.0+ epsilon)],
        [np.log(4.0 + epsilon), np.log(1.0+ epsilon)],
        [np.log(5.0 + epsilon), np.log(0.0+ epsilon)],
    ])

    np.testing.assert_array_equal(x_kernel.get(), x_exp.get())


def test_log_kernel_numpy():
    x = np.array([
        [1.0, -1.0],
        [2.0, -2.0],
        [3.0, -3.0],
        [4.0, -4.0],
        [5.0, -5.0],
        [6.0, -6.0],
    ])

    kernel = Shifted_Log_Kernel('raw')

    kernel.fit(x)

    x_kernel = kernel.transform(x)

    epsilon = Shifted_Log_Kernel.epsilon
    x_exp = np.array([
        [np.log(0.0 + epsilon), np.log(5.0+ epsilon)],
        [np.log(1.0 + epsilon), np.log(4.0+ epsilon)],
        [np.log(2.0 + epsilon), np.log(3.0+ epsilon)],
        [np.log(3.0 + epsilon), np.log(2.0+ epsilon)],
        [np.log(4.0 + epsilon), np.log(1.0+ epsilon)],
        [np.log(5.0 + epsilon), np.log(0.0+ epsilon)],
    ])

    np.testing.assert_array_equal(x_kernel, x_exp)

