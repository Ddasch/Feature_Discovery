
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


'''
Probably incorrect, need to consult literature 
'''
class Gaussian_Kernel(Abstract_Duovariate_Kernel):

    sigma=None

    def __init__(self, standardizer:str=None, sigma=1.0):
        super().__init__(standardizer)
        self.sigma = sigma

    def _fit(self, x: Union[np.ndarray, cp.ndarray]):
        self.kernel_func = cp.ElementwiseKernel(
            'float64 x, float64 y',
            'float64 z',
            'z = exp(-1 * ((x - y) *(x - y)) / (2*{sigma}))'.format(sigma=self.sigma),
            'gaussian'
        )

    def _transform(self, x: Union[np.ndarray, cp.ndarray]):
        x_ret = self.kernel_func(x[:,0], x[:, 1])
        return x_ret.reshape(-1,1)



class Polynomial_Third_Order_Kernel(Abstract_Duovariate_Kernel):

    kernel_func = None

    def _fit(self, x: Union[np.ndarray, cp.ndarray]):
        self.kernel_func = cp.ElementwiseKernel(
            'float64 x1, float64 x2',
            'float64 y1, float64 y2, float64 y3, float64 y4',
            'y1 = x1*x1*x1, y2=3*x1*x1*x2, y3=3*x1*x2*x2, y4=x2*x2*x2',
            'poly_third'
        )

    def _transform(self, x: Union[np.ndarray, cp.ndarray]):
        x_ret = self.kernel_func(x[:, 0], x[:, 1])
        x_ret = cp.column_stack(x_ret)
        return x_ret


class Polynomial_Second_Order_Kernel(Abstract_Duovariate_Kernel):

    kernel_func = None


    def _fit(self, x: Union[np.ndarray, cp.ndarray]):
        self.kernel_func = cp.ElementwiseKernel(
            'float64 x1, float64 x2',
            'float64 y1, float64 y2, float64 y3',
            'y1 = x1*x1, y2=2*x1*x2, y3=x2*x2',
            'poly_second'
        )


    def _transform(self, x: Union[np.ndarray, cp.ndarray]):
        x_ret = self.kernel_func(x[:, 0], x[:, 1])
        x_ret = cp.column_stack(x_ret)
        return x_ret



class Sigmoid_Kernel_Backwards(Abstract_Duovariate_Kernel):
    kernel_func = None

    def __init__(self, standardizer):
        super().__init__(standardizer)
        self.kernel_func = cp.ElementwiseKernel(
            'float64 dA, float64 Z',
            'float64 y',
            'y = dA * (1 / (1 + exp(-Z))) * (1 - (1 / (1 + exp(-Z))))',
            'sigmoid'
        )

    # 'y0 = exp(-1* (1/(2*1*1)) * x * x) * 1, y1 = exp(-1* (1/(2*1*1)) * x * x) * sqrt(((2*(1/(2*1*1))) / 1)) *  x'
    def _fit(self, x: Union[np.ndarray, cp.ndarray]):
        pass

    def _transform(self, x: Union[np.ndarray, cp.ndarray]):
        x_ret = self.kernel_func(x[:, 0], x[:, 1])
        return x_ret

    def sigmoid_backward(self, dA: Union[np.ndarray, cp.ndarray], Z:Union[np.ndarray, cp.ndarray]):
        x_ret = self.kernel_func(dA, Z)
        return x_ret

    '''
    def _sigmoid_backward(dA, cache):
        """
        The backward propagation for a single SIGMOID unit.
        Arguments:
        dA - post-activation gradient, of any shape
        cache - 'Z' where we store for computing backward propagation efficiently
        Returns:
        dZ - Gradient of the cost with respect to Z
        """
        Z = cache
        s = 1 / (1 + np.exp(-Z))
        dZ = dA * s * (1 - s)
        return dZ
    '''
