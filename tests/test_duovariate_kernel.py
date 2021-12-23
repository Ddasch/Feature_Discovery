import cupy as cp
import numpy as np
import pandas as pd

from featurediscovery.kernels.duovariate.duovariate_kernels import *


def test_diff_kernel():
    x = cp.array([
        [1.0, 1.0],
        [2.0, 2.0],
        [2.0, 1.0],
        [3.0, 1.0],
        [4.0, 1.0],
        [4.0, -1.0],
        [3.0, 2.0]
    ])

    kernel = Difference_Kernel('raw')

    kernel.fit(x)

    x_quad = kernel.transform(x)

    x_exp = cp.array([
        [0.0],
        [0.0],
        [1.0],
        [2.0],
        [3.0],
        [5.0],
        [1.0]
    ])

    np.testing.assert_array_equal(x_quad.get(), x_exp.get())


    #test kernel with numpy implementation

def test_diff_kernel_numpy():
    x = np.array([
        [1.0, 1.0],
        [2.0, 2.0],
        [2.0, 1.0],
        [3.0, 1.0],
        [4.0, 1.0],
        [4.0, -1.0],
        [3.0, 2.0]
    ])

    kernel = Difference_Kernel('raw')

    kernel.fit(x)

    x_diff = kernel.transform(x)

    x_exp = np.array([
        [0.0],
        [0.0],
        [1.0],
        [2.0],
        [3.0],
        [5.0],
        [1.0]
    ])

    np.testing.assert_array_equal(x_diff, x_exp)




def test_magnitude_kernel():
    x = cp.array([
        [1.0, 1.0],
        [2.0, 2.0],
        [2.0, 1.0],
        [3.0, 1.0],
        [4.0, 1.0],
        [4.0, -1.0],
    ])

    kernel = Magnitude_Kernel('raw')

    kernel.fit(x)

    x_mag = kernel.transform(x)

    x_exp = cp.array([
        [np.sqrt(2.0)],
        [np.sqrt(8.0)],
        [np.sqrt(5.0)],
        [np.sqrt(10.0)],
        [np.sqrt(17.0)],
        [np.sqrt(17.0)],
    ])

    np.testing.assert_array_equal(x_mag.get(), x_exp.get())



def test_magnitude_kernel_numpy():
    x = np.array([
        [1.0, 1.0],
        [2.0, 2.0],
        [2.0, 1.0],
        [3.0, 1.0],
        [4.0, 1.0],
        [4.0, -1.0],
    ])

    kernel = Magnitude_Kernel('raw')

    kernel.fit(x)

    x_mag = kernel.transform(x)

    x_exp = np.array([
        [np.sqrt(2.0)],
        [np.sqrt(8.0)],
        [np.sqrt(5.0)],
        [np.sqrt(10.0)],
        [np.sqrt(17.0)],
        [np.sqrt(17.0)],
    ])

    np.testing.assert_array_equal(x_mag, x_exp)

def test_gauss_kernel():
    x = cp.array([
        [1.0, 1.0],
        [2.0, 2.0],
        [2.0, 1.0],
        [3.0, 1.0],
        [4.0, 1.0],
        [4.0, -1.0],
    ])

    kernel = Gaussian_Kernel()

    kernel.fit(x)

    x_quad = kernel.transform(x)

    x_exp = cp.array([
        [1.0],
        [1.0],
        [0.6065],
        [0.1353],
        [0.0111],
        [0.0000],
    ])

    np.testing.assert_array_equal(np.around(x_quad.get(), 4), x_exp.get())



def test_poly2_kernel():
    x = cp.array([
        [1.0, 1.0],
        [2.0, 2.0],
        [2.0, 1.0],
        [3.0, 1.0],
        [4.0, 1.0],
        [4.0, -1.0],
    ])

    kernel = Polynomial_Second_Order_Kernel()

    kernel.fit(x)

    x_quad = kernel.transform(x)

    x_exp = cp.array([
        [1.0, 2*1.0, 1.0],
        [4.0, 2*4.0, 4.0],
        [4.0, 2*2.0, 1.0],
        [9., 2*3.0, 1.0],
        [16., 2*4.0, 1.0],
        [16., 2*-4.0, 1.0],
    ])

    print('exp')
    print(x_exp)
    print('result')
    print(x_quad)

    np.testing.assert_array_equal(x_quad.get(), x_exp.get())



def test_poly3_kernel():
    x = cp.array([
        [1.0, 1.0],
        [2.0, 2.0],
        [2.0, 1.0],
        [3.0, 1.0],
        [4.0, 1.0],
        [4.0, -1.0],
    ])

    kernel = Polynomial_Third_Order_Kernel('raw')

    kernel.fit(x)

    x_poly3 = kernel.transform(x)

    x_exp = cp.array([
        [1.0, 3 * 1.0, 3 * 1.0, 1.0],
        [2.0 * 2 * 2, 3 * 4.0 * 2.0, 3 * 2.0 * 4.0, 8.0],
        [2.0 * 2 * 2, 3 * 4.0 * 1., 3 * 2.0 * 1.0, 1.0],
        [3. * 3 * 3, 3 * 9.0 * 1.0, 3 * 3.0 * 1.0, 1.0],
        [4. * 4 * 4, 3 * 16.0 * 1.0, 3 * 4.0 * 1.0, 1.0],
        [4. * 4 * 4, 3 * 16.0 * -1.0, 3 * 4.0 * 1.0, -1.0],
    ])

    print('exp')
    print(x_exp)
    print('result')
    print(x_poly3)

    np.testing.assert_array_equal(x_poly3.get(), x_exp.get())



def test_poly3_kernel_numpy():
    x = np.array([
        [1.0, 1.0],
        [2.0, 2.0],
        [2.0, 1.0],
        [3.0, 1.0],
        [4.0, 1.0],
        [4.0, -1.0],
    ])

    kernel = Polynomial_Third_Order_Kernel('raw')

    kernel.fit(x)

    x_poly3 = kernel.transform(x)

    x_exp = np.array([
        [1.0, 3 * 1.0, 3 * 1.0, 1.0],
        [2.0 * 2 * 2, 3 * 4.0 * 2.0, 3 * 2.0 * 4.0, 8.0],
        [2.0 * 2 * 2, 3 * 4.0 * 1., 3 * 2.0 * 1.0, 1.0],
        [3. * 3 * 3, 3 * 9.0 * 1.0, 3 * 3.0 * 1.0, 1.0],
        [4. * 4 * 4, 3 * 16.0 * 1.0, 3 * 4.0 * 1.0, 1.0],
        [4. * 4 * 4, 3 * 16.0 * -1.0, 3 * 4.0 * 1.0, -1.0],
    ])

    print('exp')
    print(x_exp)
    print('result')
    print(x_poly3)

    np.testing.assert_array_equal(x_poly3, x_exp)