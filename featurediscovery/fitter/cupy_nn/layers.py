import cupy as cp

from kernels.abstract_kernel import Abstract_Kernel
from kernels.duovariate.duovariate_kernels import Sigmoid_Kernel_Backwards
from kernels.monovariate.monovariate_kernels import Sigmoid_Kernel


class Layer():

    W = None  #shape: [n_neuron, n_input]
    b = None
    layer_size = None
    input_size = None
    activation_kernel:Abstract_Kernel = None
    activation_kernel_backward:Abstract_Kernel = None
    activation_func = None
    learning_rate = None


    def __init__(self, input_size:int, layer_size:int, activation_func:str, learning_rate:float=0.05):

        self.layer_size = layer_size
        self.input_size = input_size

        self.W = cp.zeros((layer_size, input_size))

        self.W[:,0]=1

        self.b = cp.zeros((layer_size, 1))

        self.activation_func = activation_func
        self.learning_rate = learning_rate

        if activation_func not in ['sigmoid']:
            raise Exception('unsupported activation function')

        if activation_func == 'sigmoid':
            self.activation_kernel = Sigmoid_Kernel('dummy')
            self.activation_kernel_backward = Sigmoid_Kernel_Backwards('dummy')

    def linear_forward(self, A):
        # forward pass
        Z = cp.dot(self.W, A) + self.b


        #self.cache = (A, self.W,  self.b)
        assert (Z.shape == (self.W.shape[0], A.shape[1]))

        return Z


    def linear_activation_forward(self,A_prev):

        '''
        Arguments:
        :param A_prev:A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)

        :return:
        A -- the output of the activation function, also called the post-activation value
        cache -- a python dictionary containing "linear_cache" and "activation_cache";
                 stored for computing the backward pass efficiently
        '''

        assert A_prev.shape[0] == self.input_size

        if self.activation_func == "sigmoid":
            Z= self.linear_forward(A_prev)
            A = self.activation_kernel._transform(Z)

            #activation_cache = (Z)

        assert (A.shape == (self.W.shape[0], A_prev.shape[1]))

        self.A_prev_cache = A_prev
        self.Z_cache = Z
        self.A_cache = A

        #self.full_cache = (linear_cache, activation_cache)

        return A

    def _linear_backward(self, dZ):
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
        #A_prev, W, b = self.cache

        m = self.A_prev_cache.shape[1]

        #these are for the gradient descent optimizer for this layer
        dW = cp.dot(dZ, self.A_prev_cache.T) / m
        db = cp.squeeze(cp.sum(dZ, axis=1, keepdims=True)) / m

        #this one needs to be passed along for the next layer
        dA_prev = cp.dot(self.W.T, dZ)


        assert (dA_prev.shape == self.A_prev_cache.shape)
        assert (dW.shape == self.W.shape)
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
        #linear_cache, activation_cache = self.full_cache

        if self.activation_func == "sigmoid":
            dZ = self.sigmoid_backward(dA, self.Z_cache)

        # Shorten the code
        dA_prev= self._linear_backward(dZ)

        return dA_prev


    def recompute_weights(self):

        self.W = self.W - self.dW * self.learning_rate
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