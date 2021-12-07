

from typing import Union

import cupy as cp
import numpy as np



from featurediscovery.kernels.monovariate.abstract_monovariate import Abstract_Monovariate_Kernel


class Quadratic_Kernel(Abstract_Monovariate_Kernel):

    kernel_func = None

    def _fit(self, x: Union[np.ndarray, cp.ndarray]):
        self.kernel_func = cp.ElementwiseKernel(
            'float64 x',
            'float64 y',
            'y = x*x',
            'quadratic'
        )

    def _transform(self, x: Union[np.ndarray, cp.ndarray]):
        x_ret = self.kernel_func(x)
        return x_ret