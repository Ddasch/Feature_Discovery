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

import cupy as cp
import math

from featurediscovery.fitter.cupy_nn.activation_functions.activation_functions import *
from featurediscovery.fitter.cupy_nn.activation_functions.abstract_activation_function import AbstractActivation
from featurediscovery.fitter.cupy_nn.weight_initializer import init_2D_weights, cross_covariance, cross_corr, magnified_cross_corr

class Layer():

    W:cp.ndarray = None  #shape: [n_neuron, n_input]
    b:cp.ndarray = None
    layer_size:int = None
    input_size:int = None
    activation_func_obj:AbstractActivation = None
    learning_rate:float = None
    optimizer = None

    def __init__(self, input_size:int, layer_size:int, activation_func:str
                 , learning_rate:float=0.05
                 , weight_initializer:str='glorot'
                 , optimizer:str = 'momentum'):

        self.layer_size = layer_size
        self.input_size = input_size

        if activation_func == 'relu' and weight_initializer != 'he':
            print('WARNING! When using ReLu as activation one should use "he" as weight initializer as glorot can cause issue with ReLu')

        self.W = init_2D_weights((layer_size, input_size), input_size, layer_size, weight_initializer)
        self.b = init_2D_weights((layer_size, 1), input_size, layer_size, weight_initializer)

        self.learning_rate = learning_rate

        if activation_func not in ['sigmoid']:
            raise Exception('unsupported activation function')

        if optimizer not in ['SGD', 'momentum', 'adam']:
            raise Exception('unsupported optimizer {}'.format(optimizer))

        self.optimizer = optimizer

        #init extra matrices depending on optimizer
        if optimizer == 'momentum':
            self.VdW = cp.zeros(self.W.shape)
            self.Vdb = cp.zeros(self.b.shape)
            self.momentum_beta = 0.9

        if optimizer == 'adam':
            self.VdW = cp.zeros(self.W.shape)
            self.Vdb = cp.zeros(self.b.shape)
            self.momentum_beta = 0.9
            self.SdW = cp.zeros(self.W.shape)
            self.Sdb = cp.zeros(self.b.shape)
            self.normalization_beta = 0.5
            self.epsilon = 0.00000001

        if activation_func == 'sigmoid':
            self.activation_func_obj = SigmoidActivation()

    def _linear_forward(self, A_prev):
        '''
        Only the linear function of this layer
        :param A_prev: activations from previous layer (or input data): (size of previous layer, number of examples)
        :return: Z: the output of the linear function, aka WX+b
        '''
        # forward pass
        Z = cp.dot(self.W, A_prev) + self.b

        assert (Z.shape == (self.W.shape[0], A_prev.shape[1]))

        return Z


    def linear_activation_forward(self,A_prev):
        '''
        Complete forward activation of this layer
        :param A_prev: activations from previous layer (or input data): (size of previous layer, number of examples)
        :return: A: the output of the activation function, aka sigmoid(WX+b)
        '''

        assert A_prev.shape[0] == self.input_size

        Z = self._linear_forward(A_prev)
        A = self.activation_func_obj.activation_forward(Z)

        assert (A.shape == (self.W.shape[0], A_prev.shape[1]))

        self.A_prev_cache = A_prev
        self.Z_cache = Z
        self.A_cache = A

        return A

    def _linear_backward(self, dZ, dA_debug=None):
        '''
        Given derivative of loss w.r.t. the linear output activation, compute the derivative of the loss w.r.t
        the input activation of this layer. Also computes and caches derivatives of loss w.r.t the weights and biases
        of this layer for the optimizer to user later
        :param dZ: Gradient of the loss with respect to the linear output
        :return: Gradient of the loss with respect to the activation (of the previous layer l-1), same shape as A_prev
        '''


        #the amount of samples in the current batch
        m = self.A_prev_cache.shape[1]

        #these are for the gradient descent optimizer for this layer
        dW = cp.dot(dZ, self.A_prev_cache.T) / m
        db = cp.squeeze(cp.sum(dZ, axis=1, keepdims=True)) / m

        #this one needs to be passed along for the next layer
        dA_prev = cp.dot(self.W.T, dZ)

        assert (dA_prev.shape == self.A_prev_cache.shape)
        assert (dW.shape == self.W.shape)

        '''
        #NOTE: debug code to investigate nan values due to exploding gradients
        if cp.isnan(cp.sum(dW)):
            import numpy as np
            dZ = self.activation_func_obj.activation_backward(dA_debug, self.Z_cache)

            dZ = dZ.get()
            A_prev_cache = self.A_prev_cache.get()
            A_prev_cache = A_prev_cache.T
            tmp = np.dot(dZ, A_prev_cache)
            tmp = tmp / m


            #manual dot product for debug
            dZ_nan_indexi = np.argwhere(np.isnan(dZ))


            debug_dot_product = np.zeros((dZ.shape[0], A_prev_cache.shape[1]))
            for index_dZ in range(dZ.shape[0]):
                for index_A_prev_cache in range(A_prev_cache.shape[1]):
                    sum = 0
                    for sample_index in range(dZ.shape[1]):
                        val_dZ = dZ[index_dZ][sample_index]
                        val_A_prev_debug = A_prev_cache[sample_index][index_A_prev_cache]
                        if np.isnan(sum + val_dZ*val_A_prev_debug):
                            print('')
                        sum = sum + val_dZ*val_A_prev_debug

                    debug_dot_product[index_dZ][index_A_prev_cache] = sum

            print('')
        '''

        self.dW = dW
        self.db = db
        self.dA_prev = dA_prev

        return dA_prev

    def linear_activation_backward(self, dA):
        '''
        Given the derivative of the loss w.r.t. the output activations of this layer (including the activation function),
        compute derivative of the loss w.r.t the inputs to this layer. Also computes and caches
        derivatives of loss w.r.t the weights and biases of this layer for the optimizer to user later
        :param dA: post-activation gradient for current layer l
        :return: Gradient of the loss with respect to the activation (of the previous layer l-1), same shape as A_prev
        '''

        dZ = self.activation_func_obj.activation_backward(dA, self.Z_cache)
        dA_prev = self._linear_backward(dZ)

        return dA_prev

    def init_weights_with_data(self, X_transposed, Y_transposed, method:str):
        '''
        Make an educated guess of the parameters when doing logistic regression
        :param X_transposed: dataset of shape [n_samples, n_dims]
        :param Y_transposed: labels of shape [n_samples, n_outputs]
        :param method:
        :return:
        '''
        if method == 'corr':
            self.W = cross_corr(X_transposed, Y_transposed)
            means_X = cp.mean(X_transposed, axis=0)
            self.b = -1 * cp.sum(means_X) * cp.ones(self.b.shape)

        if method == 'magnified_corr':
            self.W = magnified_cross_corr(X_transposed, Y_transposed)
            means_X = cp.mean(X_transposed, axis=0)
            self.b = -1 * cp.sum(means_X) * cp.ones(self.b.shape)

        if method == 'corr_zero_bias':
            self.W = cross_corr(X_transposed, Y_transposed)
            self.b = self.b * 0

        if method == 'magnified_corr_zero_bias':
            self.W = magnified_cross_corr(X_transposed, Y_transposed)
            self.b = self.b * 0

        pass

    def recompute_weights(self, iteration:int):

        if self.optimizer == 'SGD':
            self.W = self.W - self.dW * self.learning_rate
            self.b = self.b - self.db * self.learning_rate

        if self.optimizer == 'momentum':
            self.VdW = self.momentum_beta * self.VdW + (1 - self.momentum_beta) * self.dW
            self.Vdb = self.momentum_beta * self.Vdb + (1 - self.momentum_beta) * self.db

            self.W = self.W - self.VdW * self.learning_rate
            self.b = self.b - self.Vdb * self.learning_rate

        if self.optimizer == 'adam':
            #momentum weight exp update
            self.VdW = self.momentum_beta * self.VdW + (1 - self.momentum_beta) * self.dW
            self.Vdb = self.momentum_beta * self.Vdb + (1 - self.momentum_beta) * self.db

            #RMS prop exp update of squared velocities
            self.SdW = self.normalization_beta * self.SdW + (1 - self.normalization_beta) * cp.multiply(self.dW,self.dW)
            self.Sdb = self.normalization_beta * self.Sdb + (1 - self.normalization_beta) * cp.multiply(self.db,self.db)

            #bias correction
            VdW_corr = self.VdW / (1 - math.pow(self.momentum_beta, iteration))
            Vdb_corr = self.Vdb / (1 - math.pow(self.momentum_beta, iteration))

            SdW_corr = self.SdW / (1 - math.pow(self.normalization_beta, iteration))
            Sdb_corr = self.Sdb / (1 - math.pow(self.normalization_beta, iteration))

            #update weights
            self.W = self.W - self.learning_rate * VdW_corr / (cp.sqrt(SdW_corr) + self.epsilon)
            self.b = self.b - self.learning_rate * Vdb_corr / (cp.sqrt(Sdb_corr) + self.epsilon)
