import cupy as cp
import numpy as np
from typing import Union
from featurediscovery.fitter.abstract_fitter import Abstract_Fitter
from featurediscovery.kernels.monovariate.monovariate_kernels import Sigmoid_Kernel


class Linear_Cupy_SGD(Abstract_Fitter):

    def _fit(self, x: Union[np.ndarray, cp.ndarray], y:Union[np.ndarray, cp.ndarray]):


        X = x.transpose()

        W = cp.zeros((1,x.shape[1]))
        W[0][0] = 1

        b = cp.zeros((1,1))

        print(x.shape)
        print(W.shape)

        n_epochs = 100
        learning_rate = 0.1

        for i in range(n_epochs):


            error = y - a

            loss = cp.absolute(error)




    sigmoid_kernel:Sigmoid_Kernel = Sigmoid_Kernel('dummy')



    def _score(self, x: Union[np.ndarray, cp.ndarray]):
        pass




