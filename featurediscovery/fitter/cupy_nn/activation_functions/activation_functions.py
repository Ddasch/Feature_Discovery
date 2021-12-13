from abc import ABC, abstractmethod

import cupy as cp

from featurediscovery.fitter.cupy_nn.activation_functions.abstract_activation_function import AbstractActivation

from featurediscovery.kernels.monovariate.monovariate_kernels import Sigmoid_Kernel

class SigmoidActivation(AbstractActivation):

    def __init__(self):
        self.sigmoid_kernel = Sigmoid_Kernel('dummy')

        #TODO turn backward pass into proper cupy kernel

    def activation_forward(self, Z:cp.ndarray):
        return self.sigmoid_kernel.transform(Z, suppres_warning=True)

    def activation_backward(self, dA:cp.ndarray, Z:cp.ndarray):
        s = 1 / (1 + cp.exp(-Z))
        dZ = dA * s * (1 - s)
        return dZ

    def recompute_weights(self):
        pass