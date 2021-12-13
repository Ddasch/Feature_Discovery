from abc import ABC, abstractmethod

import cupy as cp

from featurediscovery.fitter.cupy_nn.activation_functions.abstract_activation_function import AbstractActivation

from featurediscovery.kernels.monovariate.monovariate_kernels import Sigmoid_Kernel
from featurediscovery.kernels.duovariate.duovariate_kernels import Sigmoid_Kernel_Backwards



class SigmoidActivation(AbstractActivation):

    def __init__(self):
        self.sigmoid_kernel = Sigmoid_Kernel('dummy')

        self.sigmoid_kernel_backwards = Sigmoid_Kernel_Backwards(standardizer=None)

    def activation_forward(self, Z:cp.ndarray):
        return self.sigmoid_kernel.transform(Z, suppres_warning=True)

    def activation_backward(self, dA:cp.ndarray, Z:cp.ndarray):
        '''
        s = 1 / (1 + cp.exp(-Z))
        dZ = dA * s * (1 - s)
        return dZ
        '''
        return self.sigmoid_kernel_backwards.sigmoid_backward(dA,Z)

    def recompute_weights(self):
        pass