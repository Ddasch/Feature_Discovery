import cupy as cp
import numpy as np
import pandas as pd

from featurediscovery.kernels.duovariate.duovariate_kernels import Difference_Kernel


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