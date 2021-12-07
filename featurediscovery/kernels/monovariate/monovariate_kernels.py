

from typing import Union

import cupy as cp
import numpy as np



from featurediscovery.kernels.monovariate.abstract_monovariate import Abstract_Monovariate_Kernel


class Quadratic_Kernel(Abstract_Monovariate_Kernel):

    kernel_func = None
    #'y0 = exp(-1* (1/(2*1*1)) * x * x) * 1, y1 = exp(-1* (1/(2*1*1)) * x * x) * sqrt(((2*(1/(2*1*1))) / 1)) *  x'
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

