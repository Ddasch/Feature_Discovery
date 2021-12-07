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

    kernel = Quadratic_Kernel()

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