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
    ])

    kernel = Difference_Kernel()

    kernel.fit(x)

    x_quad = kernel.transform(x)

    x_exp = cp.array([
        [0.0],
        [0.0],
        [1.0],
        [2.0],
        [3.0],
        [5.0],
    ])

    np.testing.assert_array_equal(x_quad.get(), x_exp.get())



def test_magnitude_kernel():
    x = cp.array([
        [1.0, 1.0],
        [2.0, 2.0],
        [2.0, 1.0],
        [3.0, 1.0],
        [4.0, 1.0],
        [4.0, -1.0],
    ])

    kernel = Magnitude_Kernel()

    kernel.fit(x)

    x_quad = kernel.transform(x)

    x_exp = cp.array([
        [np.sqrt(2.0)],
        [np.sqrt(8.0)],
        [np.sqrt(5.0)],
        [np.sqrt(10.0)],
        [np.sqrt(17.0)],
        [np.sqrt(17.0)],
    ])

    np.testing.assert_array_equal(x_quad.get(), x_exp.get())