
from typing import Union, List

import cupy as cp
import numpy as np
import pandas as pd

from featurediscovery.kernels.duovariate.abstract_duovariate import Abstract_Duovariate_Kernel
from sklearn.preprocessing import PolynomialFeatures


SUPPORTED_DUOVARIATE_KERNELS = ['difference', 'magnitude', 'poly3', 'poly2']


def get_duo_kernel(kernel_name:str, standardizer:str) -> Abstract_Duovariate_Kernel:
    if kernel_name not in SUPPORTED_DUOVARIATE_KERNELS:
        raise Exception('Unknown kernel {}. Supported kernels are {}'.format(kernel_name, SUPPORTED_DUOVARIATE_KERNELS))

    if kernel_name == 'poly2':
        return Polynomial_Second_Order_Kernel(standardizer)

    if kernel_name == 'poly3':
        return Polynomial_Third_Order_Kernel(standardizer)

    if kernel_name == 'difference':
        return Difference_Kernel(standardizer)

    if kernel_name == 'magnitude':
        return Magnitude_Kernel(standardizer)


difference_kernel_singleton = None
class Difference_Kernel(Abstract_Duovariate_Kernel):

    def __init__(self, standardizer:str):
        super().__init__(standardizer)

        global difference_kernel_singleton
        if difference_kernel_singleton is None:
            difference_kernel_singleton = cp.ElementwiseKernel(
                'float64 x, float64 y',
                'float64 z',
                'z = x - y',
                'difference'
            )

        self.kernel_func = difference_kernel_singleton


    def _fit(self, x: Union[np.ndarray, cp.ndarray]):
        pass

    def _transform(self, x: Union[np.ndarray, cp.ndarray]):
        if self.api == 'cupy':
            x_ret = self.kernel_func(x[:,0], x[:, 1])
            return x_ret.reshape(-1,1)
        else:
            return (x[:,0] - x[:,1]).reshape(-1,1)

    def _get_kernel_feature_names(self, f1: str, f2: str):
        return ['{} - {}'.format(f1,f2)]

    def get_kernel_name(self):
        return 'Difference {} {}'.format(self.kernel_input_features[0], self.kernel_input_features[1])

    def get_kernel_type(self):
        return 'Difference'

magnitude_kernel_singleton = None
class Magnitude_Kernel(Abstract_Duovariate_Kernel):
    def __init__(self, standardizer:str):
        super().__init__(standardizer)

        global magnitude_kernel_singleton
        if magnitude_kernel_singleton is None:
            magnitude_kernel_singleton = cp.ElementwiseKernel(
                'float64 x, float64 y',
                'float64 z',
                'z = sqrt((x * x) + (y * y))',
                'magnitude'
            )

        self.kernel_func = magnitude_kernel_singleton

    def _fit(self, x: Union[np.ndarray, cp.ndarray]):
        pass

    def _transform(self, x: Union[np.ndarray, cp.ndarray]):
        if self.api == 'cupy':
            x_ret = self.kernel_func(x[:,0], x[:, 1])
            return x_ret.reshape(-1,1)
        else:
            x_squared = np.multiply(x[:,0], x[:,0]) + np.multiply(x[:,1], x[:,1])
            x_mag = np.sqrt(x_squared)
            return x_mag.reshape(-1,1)

    def _get_kernel_feature_names(self, f1: str, f2: str):
        return ['||{},{}}}'.format(f1,f2)]

    def get_kernel_name(self):
        return 'Magnitue {} {}'.format(self.kernel_input_features[0], self.kernel_input_features[1])

    def get_kernel_type(self):
        return 'Magnitude'

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


polynomial_3_singleton = None
class Polynomial_Third_Order_Kernel(Abstract_Duovariate_Kernel):

    kernel_func = None
    def __init__(self, standardizer:str):
        super().__init__(standardizer)
        global polynomial_3_singleton
        if polynomial_3_singleton is None:
            polynomial_3_singleton = cp.ElementwiseKernel(
                'float64 x1, float64 x2',
                'float64 y1, float64 y2, float64 y3, float64 y4',
                'y1 = x1*x1*x1, y2=3*x1*x1*x2, y3=3*x1*x2*x2, y4=x2*x2*x2',
                'poly_third'
            )

        self.kernel_func = polynomial_3_singleton

    def _fit(self, x: Union[np.ndarray, cp.ndarray]):
        pass

    def _transform(self, x: Union[np.ndarray, cp.ndarray]):
        if self.api == 'cupy':
            x_ret = self.kernel_func(x[:, 0], x[:, 1])
            x_ret = cp.column_stack(x_ret)
            return x_ret
        else:
            x_f1_3 = np.multiply(np.multiply(x[:, 0],x[:, 0]), x[:, 0])
            x_3_f1_2_f2 = 3 * np.multiply(np.multiply(x[:, 0], x[:, 0]), x[:, 1])
            x_3_f1_f2_2 = 3 * np.multiply(np.multiply(x[:, 0], x[:, 1]), x[:, 1])
            x_f2_3 = np.multiply(np.multiply(x[:, 1], x[:, 1]), x[:, 1])
            x_ret = np.column_stack((x_f1_3,x_3_f1_2_f2,x_3_f1_f2_2,x_f2_3))
            return x_ret

    def _get_kernel_feature_names(self, f1:str, f2:str):
        return ['{}^3'.format(f1)
            , '3*{}^2*{}'.format(f1, f2)
            , '3*{}*{}^2'.format(f1, f2)
            , '{}^3'.format(f2)]

    def get_kernel_name(self) -> str:
        return 'Polynomial3 {x1} {x2} {std}'.format(x1=self.kernel_input_features[0], x2=self.kernel_input_features[1], std=self.standardizer.get_standardizer_name())

    def get_kernel_type(self) -> str:
        return 'Polynomial3'


polynomial_2_singleton = None
class Polynomial_Second_Order_Kernel(Abstract_Duovariate_Kernel):

    kernel_func = None
    poly:PolynomialFeatures = None

    def _fit(self, x: Union[np.ndarray, cp.ndarray]):
        use_cupy = type(x) == cp.ndarray
        if use_cupy:
            global polynomial_2_singleton
            if polynomial_2_singleton is None:
                polynomial_2_singleton = cp.ElementwiseKernel(
                    'float64 x1, float64 x2',
                    'float64 y1, float64 y2, float64 y3',
                    'y1 = x1*x1, y2=2*x1*x2, y3=x2*x2',
                    'poly_second'
                )

            self.kernel_func = polynomial_2_singleton
        else:
            self.poly = PolynomialFeatures(2)


    def _transform(self, x: Union[np.ndarray, cp.ndarray]):

        if self.api == 'cupy':
            x_ret = self.kernel_func(x[:, 0], x[:, 1])
            x_ret = cp.column_stack(x_ret)
            return x_ret
        else:
            return self.poly.fit_transform(x)[:,3:]


    def _get_kernel_feature_names(self, f1:str, f2:str):
        return ['{}^2'.format(f1)
            , '{}*{}'.format(f1, f2)
            , '{}^2'.format(f2)]

    def get_kernel_name(self) -> str:
        return 'Polynomial2 {x1} {x2} {std}'.format(x1=self.kernel_input_features[0], x2=self.kernel_input_features[1], std=self.standardizer.get_standardizer_name())

    def get_kernel_type(self) -> str:
        return 'Polynomial2'

sigmoid_kernel_backward_singleton = None
class Sigmoid_Kernel_Backwards(Abstract_Duovariate_Kernel):
    kernel_func = None

    def __init__(self, standardizer):
        super().__init__(standardizer)
        global sigmoid_kernel_backward_singleton
        if sigmoid_kernel_backward_singleton is None:
            sigmoid_kernel_backward_singleton = cp.ElementwiseKernel(
                'float64 dA, float64 Z',
                'float64 y',
                'y = dA * (1 / (1 + exp(-Z))) * (1 - (1 / (1 + exp(-Z))))',
                'sigmoidbackprop'
            )

        self.kernel_func = sigmoid_kernel_backward_singleton

    # 'y0 = exp(-1* (1/(2*1*1)) * x * x) * 1, y1 = exp(-1* (1/(2*1*1)) * x * x) * sqrt(((2*(1/(2*1*1))) / 1)) *  x'
    def _fit(self, x: Union[np.ndarray, cp.ndarray]):
        pass

    def _transform(self, x: Union[np.ndarray, cp.ndarray]):
        if self.api == 'cupy':
            x_ret = self.kernel_func(x[:, 0], x[:, 1])
            return x_ret
        if self.api == 'numpy':
            raise Exception('not yet implemented')

    def sigmoid_backward(self, dA: Union[np.ndarray, cp.ndarray], Z:Union[np.ndarray, cp.ndarray]):
        x_ret = self.kernel_func(dA, Z)
        return x_ret


    def get_kernel_name(self):
        return 'Sigmoid Backprop dA={} Z={}'.format(self.kernel_input_features[0], self.kernel_input_features[1])

    #def get_kernel_feature_names(self, input_features:List[str]=None):
    #    return ['dA_prev']

    def _get_kernel_feature_names(self, f1: str, f2: str):
        return ['dA_prev']

    def get_kernel_type(self) -> str:
        return 'Sigmoid Backprop'

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



class RFF_Kernel(Abstract_Duovariate_Kernel):

    W = None
    b = None
    n_transforms: int = 100

    def _fit(self, x: Union[np.ndarray, cp.ndarray]):
        if self.api == 'cupy':
            #self.W = cp.random.randn(self.n_transforms, x.shape[1])
            #self.b = cp.random.randn(self.n_transforms, 1) * 2 * cp.pi
            self.W = cp.random.randn(x.shape[1],self.n_transforms)
            self.b = cp.random.randn(1, self.n_transforms) * 2 * cp.pi
        else:
            #self.W = np.random.randn(self.n_transforms, x.shape[1])
            #self.b = np.random.randn(self.n_transforms, 1) * 2 * np.pi
            self.W = np.random.randn(x.shape[1], self.n_transforms)
            self.b = np.random.randn(1, self.n_transforms) * 2 * np.pi


    def _transform(self, x: Union[np.ndarray, cp.ndarray]):

        if self.api == 'cupy':
            #Z = cp.dot(self.W, x.transpose()) + self.b
            #A = cp.cos(Z)
            Z = cp.dot(x, self.W) + self.b
            A = cp.cos(Z)
        else:
            #Z = np.dot(self.W, x.transpose()) + self.b
            #A = np.cos(Z)
            Z = np.dot(x, self.W) + self.b
            A = np.cos(Z)

        return A


    def _get_kernel_feature_names(self, f1:str, f2:str):
        names = []
        for i in range(self.n_transforms):
            names.append('RFF {}'.format(i))

        return names

    def get_kernel_name(self) -> str:
        return 'RFF {x1} {x2} {std}'.format(x1=self.kernel_input_features[0], x2=self.kernel_input_features[1], std=self.standardizer.get_standardizer_name())

    def get_kernel_type(self) -> str:
        return 'RFF'