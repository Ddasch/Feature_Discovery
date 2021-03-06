# Copyright 2021-2022 by Frederik Christiaan Schadd
# All rights reserved
#
# Licensed under the GNU Lesser General Public License version 2.1 ;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://opensource.org/licenses/LGPL-2.1
#
# or consult the LICENSE file included in the project.



from typing import Union

import cupy as cp
import numpy as np
import pandas as pd


from featurediscovery.kernels.monovariate.abstract_monovariate import Abstract_Monovariate_Kernel


SUPPORTED_MONOVARIATE_KERNELS = ['quadratic', 'gaussian_approx', 'sigmoid', 'log', 'log_shift']


def get_mono_kernel(name:str, standardizer:str) -> Abstract_Monovariate_Kernel:
    if name not in SUPPORTED_MONOVARIATE_KERNELS:
        raise Exception('unsupported kernel')

    if name == 'quadratic':
        return Quadratic_Kernel(standardizer)

    if name == 'gaussian_approx':
        return Gaussian_Kernel_Taylor_Aprox(standardizer)

    if name == 'sigmoid':
        return Sigmoid_Kernel(standardizer)

    if name == 'log':
        return Log_Kernel(standardizer)

    if name == 'log_shift':
        return Shifted_Log_Kernel(standardizer)



quadratic_cupy_kernel_singleton = None
class Quadratic_Kernel(Abstract_Monovariate_Kernel):

    kernel_func = None

    def __init__(self, standardizer:str):
        super().__init__(standardizer)
        global quadratic_cupy_kernel_singleton
        if quadratic_cupy_kernel_singleton is None:
            quadratic_cupy_kernel_singleton = cp.ElementwiseKernel(
                'float64 x',
                'float64 y',
                'y = x*x',
                'quadratic'
            )
        self.kernel_func = quadratic_cupy_kernel_singleton



    #'y0 = exp(-1* (1/(2*1*1)) * x * x) * 1, y1 = exp(-1* (1/(2*1*1)) * x * x) * sqrt(((2*(1/(2*1*1))) / 1)) *  x'
    def _fit(self, x: Union[np.ndarray, cp.ndarray]):
        pass

    def _transform(self, x: Union[np.ndarray, cp.ndarray]):
        use_cupy = type(x) == cp.ndarray
        if use_cupy:
            x_ret = self.kernel_func(x)
        else:
            x_ret = np.multiply(x,x)
        return x_ret

    ''' 
    def apply(self, df:pd.DataFrame):
        if not self.finalized:
            raise Exception('Attempting to apply kernel that has not been finalized yet')

        X = df[self.features].to_numpy(dtype=np.float64)

        X_square = self.transform(X)

        k_feat_name = self.features[0] + '^2'

        df[k_feat_name] = X_square

        self.kernel_features = [k_feat_name]

        return df
    '''

    def get_kernel_name(self):
        return 'Quadratic {} {}'.format(self.kernel_input_features[0], self.standardizer.get_standardizer_name())

    def _get_kernel_feature_names(self, f1):
        return [f1 + '^2']

    def get_kernel_type(self) -> str:
        return 'Quadratic'


class Gaussian_Kernel_Taylor_Aprox(Abstract_Monovariate_Kernel):
    '''
    https://czxttkl.com/2020/04/27/revisit-gaussian-kernel/
    '''
    kernel_func = None
    degree = None
    sigma = None

    def __init__(self, standardizer:str=None, degree=1, sigma=1):
        super().__init__(standardizer)

        if degree < 1:
            raise Exception('degree must be positive')

        self.degree = degree
        self.sigma = sigma

    def _get_gamma(self, sigma):
        return '(1/(2*{sigma}*{sigma}))'.format(sigma=sigma)

    def _get_taylor_base_factor(self, sigma):
        return 'exp(-1* {gamma} * x * x)'.format(gamma=self._get_gamma(sigma))

    def _get_faculty(self, number):
        result = '{}'.format(number)
        for i in range(number - 1):
            f = i +1
            result = result + ' * {}'.format(f)

        return result

    def _get_derivative_term(self, derivative_order:int=0, sigma=1):

        if derivative_order < 0:
            raise Exception('cannot be negative')

        if derivative_order == 0:
            return '1'
        else:

            fac_numerator_element = '(2*{gamma})'.format(gamma=self._get_gamma(sigma))

            fac_numerator = fac_numerator_element
            for i in range(derivative_order - 1):
                fac_numerator = fac_numerator + ' * ' + fac_numerator_element

            fac_divisor = self._get_faculty(derivative_order)

            factor_complete = 'sqrt(const1 * {numerator} / {dividor} )'.format(numerator=fac_numerator, dividor=fac_divisor)

            x_derived = ' x'
            for i in range(derivative_order - 1):
                x_derived = x_derived + ' * x'

            taylor_term = factor_complete \
                          + ' * ' + x_derived

            return taylor_term



    def _fit(self, x: Union[np.ndarray, cp.ndarray]):


        outputs = ''
        functions = ''

        for i in range(self.degree):
            outputs = outputs + 'float64 y{}, '.format(i)

            base_fac = self._get_taylor_base_factor(self.sigma)
            derivative_term = self._get_derivative_term(derivative_order=i)

            equation = 'y{i} = {base_fac} * {derivative_term}'.format(i=i, base_fac=base_fac, derivative_term=derivative_term)
            functions = functions + equation + ', '

        outputs = outputs[:-2]
        functions = functions[:-2]

        self.kernel_func = cp.ElementwiseKernel(
            'float64 x, float64 const1',
            outputs,
            functions,
            'gaussian_approx'
        )

    def _transform(self, x: Union[np.ndarray, cp.ndarray]):
        const1 = cp.ones(x.shape)

        x_ret = self.kernel_func(x, const1)
        return x_ret


class Sigmoid_Kernel(Abstract_Monovariate_Kernel):

    kernel_func = None

    def __init__(self, standardizer:str=None):
        super().__init__(standardizer)
        self.kernel_func = cp.ElementwiseKernel(
            'float64 x',
            'float64 y',
            'y = 1/(1 + exp(-1*x))',
            'sigmoid'
        )


    #'y0 = exp(-1* (1/(2*1*1)) * x * x) * 1, y1 = exp(-1* (1/(2*1*1)) * x * x) * sqrt(((2*(1/(2*1*1))) / 1)) *  x'
    def _fit(self, x: Union[np.ndarray, cp.ndarray]):
        pass

    def _transform(self, x: Union[np.ndarray, cp.ndarray]):
        if type(x) == cp.ndarray:
            x_ret = self.kernel_func(x)
            return x_ret
        else:
            x_ret = 1 / (1 + np.exp(-1*x))
            return x_ret

    '''
    def apply(self, df:pd.DataFrame):
        if not self.finalized:
            raise Exception('Attempting to apply kernel that has not been finalized yet')

        X = df[self.kernel_input_features].to_numpy(dtype=np.float64)

        X_square = self.transform(X)

        k_feat_name = 'sigmoid(' + self.kernel_input_features[0] + ')'

        df[k_feat_name] = X_square

        self.kernel_features = [k_feat_name]

        return df
    '''

    def get_kernel_name(self):
        return 'Sigmoid {} {}'.format(self.kernel_input_features[0], self.standardizer.get_standardizer_name())

    def _get_kernel_feature_names(self, f1):
        return ['sig({})'.format(f1)]

    def get_kernel_type(self) -> str:
        return 'sigmoid'




class Shifted_Log_Kernel(Abstract_Monovariate_Kernel):
    kernel_func = None
    minima = None
    epsilon = 0.0001

    def __init__(self, standardizer: str):
        super().__init__(standardizer)

    def _fit(self, x: Union[np.ndarray, cp.ndarray]):
        self.minima = x.min(axis=0)

    def _transform(self, x: Union[np.ndarray, cp.ndarray]):

        if self.api == 'cupy':
            x_ret = x - self.minima + self.epsilon
            x_ret = cp.log(x_ret)
        else:
            x_ret = x - self.minima + self.epsilon
            x_ret = np.log(x_ret)
        return x_ret


    def get_kernel_name(self):
        return 'Shifted Log {} {}'.format(self.kernel_input_features[0], self.standardizer.get_standardizer_name())

    def _get_kernel_feature_names(self, f1):
        return ['shift_log(' +f1 + ')']

    def get_kernel_type(self) -> str:
        return 'Shifted Log'


class Log_Kernel(Abstract_Monovariate_Kernel):
    kernel_func = None
    epsilon = 0.0001

    def __init__(self, standardizer: str):
        super().__init__(standardizer)

    def _fit(self, x: Union[np.ndarray, cp.ndarray]):
        pass

    def _transform(self, x: Union[np.ndarray, cp.ndarray]):

        if self.api == 'cupy':
            x_ret = cp.maximum(x, cp.ones(x.shape)*self.epsilon)
            x_ret = cp.log(x_ret)
        else:
            x_ret = np.maximum(x, np.ones(x.shape)*self.epsilon)
            x_ret = np.log(x_ret)
        return x_ret


    def get_kernel_name(self):
        return 'Log {} {}'.format(self.kernel_input_features[0], self.standardizer.get_standardizer_name())

    def _get_kernel_feature_names(self, f1):
        return ['log(' +f1 + ')']

    def get_kernel_type(self) -> str:
        return 'Log'