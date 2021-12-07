import cupy as cp
import numpy as np
import pandas as pd

from featurediscovery.kernels.monovariate.monovariate_kernels import Quadratic_Kernel



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
