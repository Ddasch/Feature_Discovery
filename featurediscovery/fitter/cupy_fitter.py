import cupy as cp
import numpy as np
from typing import Union
from featurediscovery.fitter.abstract_fitter import Abstract_Fitter
from featurediscovery.kernels.monovariate.monovariate_kernels import Sigmoid_Kernel
from featurediscovery.kernels.duovariate.duovariate_kernels import Sigmoid_Kernel_Backwards
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

    W = None  #shape: [n_neuron, n_input]
    b = None
    layer_size = None
    input_size = None
    activation_kernel:Abstract_Kernel = None
    activation_kernel_backward:Abstract_Kernel = None
    activation_func = None
    learning_rate = 0.1


    def __init__(self, input_size:int, layer_size:int, activation_func:str):

        self.layer_size = layer_size
        self.input_size = input_size

        self.W = cp.zeros((layer_size, input_size))

        self.W[:,0]=1

        self.b = cp.zeros((layer_size, 1))

        self.activation_func = activation_func
        if activation_func == 'sigmoid':
            self.activation_kernel = Sigmoid_Kernel('dummy')
            self.activation_kernel_backward = Sigmoid_Kernel_Backwards('dummy')

    def linear_forward(self, A):
        # forward pass
        Z = cp.dot(self.W, A) + self.b


        self.cache = (A, self.W,  self.b)
        assert (Z.shape == (self.W.shape[0], A.shape[1]))

        return Z, self.cache


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

        self.full_cache = (linear_cache, activation_cache)

        return A, self.full_cache

    def linear_backward(self,dZ):
        """
        Implement the linear portion of backward propagation for a single layer (layer l)

        Arguments:
        dZ -- Gradient of the cost with respect to the linear output (of current layer l)
        cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        A_prev, W, b = self.cache
        m = A_prev.shape[1]

        #these are for the gradient descent optimizer for this layer
        dW = np.dot(dZ, A_prev.T) / m
        db = np.squeeze(np.sum(dZ, axis=1, keepdims=True)) / m

        #this one needs to be passed along for the next layer
        dA_prev = np.dot(W.T, dZ)


        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        #assert (isinstance(db, np.float64))

        self.dW = dW
        self.db = db
        self.dA_prev = dA_prev

        return dA_prev, dW, db

    def linear_activation_backward(self, dA):
        """
        Implement the backward propagation for the LINEAR->ACTIVATION layer.

        Arguments:
        dA -- post-activation gradient for current layer l
        cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        linear_cache, activation_cache = self.full_cache

        if self.activation_func == "sigmoid":
            ### START CODE HERE ### (≈ 2 lines of code)
            dZ = self.sigmoid_backward(dA, activation_cache)
            ### END CODE HERE ###

        # Shorten the code
        dA_prev, dW, db = self.linear_backward(dZ)

        return dA_prev, dW, db


    def recompute_weights(self):

        self.W  = self.W - self.dW * self.learning_rate
        self.b = self.b - self.db * self.learning_rate


    def sigmoid_backward(self, dA, cache):
        """
        The backward propagation for a single SIGMOID unit.
        Arguments:
        dA - post-activation gradient, of any shape
        cache - 'Z' where we store for computing backward propagation efficiently
        Returns:
        dZ - Gradient of the cost with respect to Z
        """
        Z = cache
        s = 1/(1+cp.exp(-Z))
        dZ = dA * s * (1-s)
        return dZ

class CrossEntropyCost():

    epsilon = 1e-12

    def compute_cost(self, AL, Y):
        """
            Implement the cost function defined by equation (7).

            Arguments:
            AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
            Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

            Returns:
            cost -- cross-entropy cost
            """

        m = Y.shape[1]

        # Compute loss from aL and y.
        AL = cp.clip(AL, self.epsilon, 1-self.epsilon)
        #log_al = cp.log(AL)
        #log_1m_al = cp.log(1-AL)

        cost = (-1 / m) * cp.sum(cp.multiply(Y, cp.log(AL)) + cp.multiply(1 - Y, cp.log(1 - AL)))

        cost = cp.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
        assert (cost.shape == ())

        return cost

    def loss_backward(self, AL, Y):
        #source: http://www.adeveloperdiary.com/data-science/deep-learning/neural-network-with-softmax-in-python/
        return cp.subtract(AL,Y)


class SimpleModel():

    activation_func = None

    def __init__(self, activation:str='sigmoid'):
        self.activation_func = activation

    def fit(self, x:cp.ndarray, y:cp.ndarray):
        #shape x: [n_sample, n_feature], standard format what you get out of a df


        X = x.transpose()
        Y = y.transpose()

        #shape X: [n_feature, n_sample] as layer expects other way around

        layer =Layer(input_size=X.shape[0], layer_size=1, activation_func=self.activation_func)

        cross_e = CrossEntropyCost()

        for i in range(20):
            A, full_cache = layer.linear_activation_forward(X)

            cost_val = cross_e.compute_cost(A, Y)

            print('current cost: {}'.format(cost_val))

            dL_A = cross_e.loss_backward(A,Y)

            dA_prev, dW, db = layer.linear_backward(dL_A)

            layer.recompute_weights()

            print('')

        self.layer = layer
        self.cross_e = cross_e

    def score(self, x:cp.ndarray,):
        X = x.transpose()
        A, full_cache = self.layer.linear_activation_forward(X)

        A = A.transpose()

        A_label = A.copy()
        A_label[A_label>0.5]=1
        A_label[A_label <=0.5] = 0

        return A_label, A