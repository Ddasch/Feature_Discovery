import cupy as cp
import numpy as np
from typing import Union
from featurediscovery.fitter.abstract_fitter import Abstract_Fitter
from featurediscovery.kernels.monovariate.monovariate_kernels import Sigmoid_Kernel
from featurediscovery.kernels.abstract_kernel import Abstract_Kernel




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




class Layer():

    W = None
    b = None
    layer_size = None
    input_size = None
    activation_kernel:Abstract_Kernel = None
    activation_func = None
    def __init__(self, input_size:int, layer_size:int, activation_func:str):

        self.layer_size = layer_size
        self.input_size = input_size

        self.W = cp.zeros((layer_size, input_size))

        self.W[:,0]=1

        self.b = cp.zeros((layer_size, 1))

        self.activation_func = activation_func
        if activation_func == 'sigmoid':
            self.activation_kernel = Sigmoid_Kernel('dummy')

    def linear_forward(self, A):
        # forward pass
        Z = cp.dot(self.W, A) + self.b

        cache = (self.W, A, self.b)
        assert (Z.shape == (self.W.shape[0], A.shape[1]))

        return Z, cache


    def linear_activation_forward(self,A_prev):

        '''
        Arguments:
        :param A_prev:A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
        :param W: W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        :param b: b -- bias vector, numpy array of shape (size of the current layer, 1)
        :param activation:  activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"




        :return:
        A -- the output of the activation function, also called the post-activation value
        cache -- a python dictionary containing "linear_cache" and "activation_cache";
                 stored for computing the backward pass efficiently
        '''

        assert A_prev.shape[0] == self.input_size

        if self.activation_func == "sigmoid":
            Z, linear_cache = self.linear_forward(A_prev)
            A = self.activation_kernel._transform(Z)

            activation_cache = (Z)

        assert (A.shape == (self.W.shape[0], A_prev.shape[1]))

        cache = (linear_cache, activation_cache)

        return A, cache