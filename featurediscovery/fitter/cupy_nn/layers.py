import cupy as cp


from featurediscovery.fitter.cupy_nn.activation_functions.activation_functions import *
from featurediscovery.fitter.cupy_nn.activation_functions.abstract_activation_function import AbstractActivation
from featurediscovery.fitter.cupy_nn.weight_initializer import init_2D_weights

class Layer():

    W:cp.ndarray = None  #shape: [n_neuron, n_input]
    b:cp.ndarray = None
    layer_size:int = None
    input_size:int = None
    activation_func_obj:AbstractActivation = None
    learning_rate:float = None


    def __init__(self, input_size:int, layer_size:int, activation_func:str
                 , learning_rate:float=0.05
                 , weight_initializer:str='glorot'):

        self.layer_size = layer_size
        self.input_size = input_size

        if activation_func == 'relu' and weight_initializer != 'he':
            print('WARNING! When using ReLu as activation one should use "he" as weight initializer as glorot can cause issue with ReLu')

        self.W = init_2D_weights((layer_size, input_size), input_size, layer_size, weight_initializer)
        self.b = init_2D_weights((layer_size, 1), input_size, layer_size, weight_initializer)

        self.learning_rate = learning_rate

        if activation_func not in ['sigmoid']:
            raise Exception('unsupported activation function')

        if activation_func == 'sigmoid':
            self.activation_func_obj = SigmoidActivation()

    def _linear_forward(self, A):
        # forward pass
        Z = cp.dot(self.W, A) + self.b

        assert (Z.shape == (self.W.shape[0], A.shape[1]))

        return Z


    def linear_activation_forward(self,A_prev):
        '''
        Arguments:
        :param A_prev:A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)

        :return:
        A -- the output of the activation function, aka sigmoid(WX+b)
        '''

        assert A_prev.shape[0] == self.input_size

        Z = self._linear_forward(A_prev)
        A = self.activation_func_obj.activation_forward(Z)

        assert (A.shape == (self.W.shape[0], A_prev.shape[1]))

        self.A_prev_cache = A_prev
        self.Z_cache = Z
        self.A_cache = A

        return A

    def _linear_backward(self, dZ):
        """
        Given derivative of loss w.r.t. the linear output activation, compute the derivative of the loss w.r.t
        the input activation of this layer. Also computes and caches derivatives of loss w.r.t the weights and biases
        of this layer for the optimizer to user later

        Arguments:
        dZ -- Gradient of the loss with respect to the linear output

        Returns:
        dA_prev -- Gradient of the loss with respect to the activation (of the previous layer l-1), same shape as A_prev
        """

        #the amount of samples in the current batch
        m = self.A_prev_cache.shape[1]

        #these are for the gradient descent optimizer for this layer
        dW = cp.dot(dZ, self.A_prev_cache.T) / m
        db = cp.squeeze(cp.sum(dZ, axis=1, keepdims=True)) / m

        #this one needs to be passed along for the next layer
        dA_prev = cp.dot(self.W.T, dZ)

        assert (dA_prev.shape == self.A_prev_cache.shape)
        assert (dW.shape == self.W.shape)

        self.dW = dW
        self.db = db
        self.dA_prev = dA_prev

        return dA_prev, dW, db

    def linear_activation_backward(self, dA):
        """
        Given the derivative of the loss w.r.t. the output activations of this layer (including the activation function),
        compute derivative of the loss w.r.t the inputs to this layer. Also computes and caches
        derivatives of loss w.r.t the weights and biases of this layer for the optimizer to user later

        Arguments:
        dA -- post-activation gradient for current layer l
        Returns:
        dA_prev -- Gradient of the loss with respect to the activation (of the previous layer l-1), same shape as A_prev
        """
        dZ = self.activation_func_obj.activation_backward(dA, self.Z_cache)
        dA_prev = self._linear_backward(dZ)

        return dA_prev


    def recompute_weights(self):

        self.W = self.W - self.dW * self.learning_rate
        self.b = self.b - self.db * self.learning_rate
