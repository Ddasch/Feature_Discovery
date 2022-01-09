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



from abc import ABC, abstractmethod

import cupy as cp



class AbstractActivation(ABC):

    @abstractmethod
    def activation_forward(self, Z:cp.ndarray):
        '''
        Forward step of the activation function
        :param Z: inputs to this activation function, most likely the linear activations of the neurons
        :return: A: output of this activation function
        '''
        pass

    @abstractmethod
    def activation_backward(self, dA:cp.ndarray, Z:cp.ndarray):
        '''
        Backprop step of the activation function
        :param dA: post-activation gradient, of any shape
        :param Z: the cached Z values that were the input to this activation function during the forward pass
        :return: dZ: derivative of cost w.r.t. the inputs of this activation function
        '''
        pass

    @abstractmethod
    def recompute_weights(self):
        '''
        If this function itself has weights, update these during the gradient descent step
        :return:
        '''
        pass

