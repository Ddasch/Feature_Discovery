
from typing import Union

import cupy as cp
import numpy as np

from featurediscovery.kernels.duovariate.abstract_duovariate import Abstract_Duovariate_Kernel


class Difference_Kernel(Abstract_Duovariate_Kernel):

    def _fit(self, x: Union[np.ndarray, cp.ndarray]):
        self.kernel_func = cp.ElementwiseKernel(
            'float64 x, float64 y',
            'float64 z',
            'z = x - y',
            'difference'
        )

    def _transform(self, x: Union[np.ndarray, cp.ndarray]):
        x_ret = self.kernel_func(x[:,0], x[:, 1])
        return x_ret.reshape(-1,1)


class Magnitude_Kernel(Abstract_Duovariate_Kernel):
    def _fit(self, x: Union[np.ndarray, cp.ndarray]):
        self.kernel_func = cp.ElementwiseKernel(
            'float64 x, float64 y',
            'float64 z',
            'z = sqrt((x * x) + (y * y))',
            'magnitude'
        )

    def _transform(self, x: Union[np.ndarray, cp.ndarray]):
        x_ret = self.kernel_func(x[:,0], x[:, 1])
        return x_ret.reshape(-1,1)