import cupy as cp

from fitter.cupy_nn.costs import CrossEntropyCost
from fitter.cupy_nn.layers import Layer
from typing import List

class SimpleModel():

    activation_func = None

    def __init__(self, activation:str='sigmoid'):
        self.activation_func = activation

    def fit(self, x:cp.ndarray, y:cp.ndarray):
        #shape x: [n_sample, n_feature], standard format what you get out of a df


        X = x.transpose()
        Y = y.transpose()

        #shape X: [n_feature, n_sample] as layer expects other way around

        layer = Layer(input_size=X.shape[0], layer_size=1, activation_func=self.activation_func)

        cross_e = CrossEntropyCost()

        for i in range(20):
            A, full_cache = layer.linear_activation_forward(X)

            cost_val = cross_e.compute_cost(A, Y)

            print('current cost: {}'.format(cost_val))

            dL_A = cross_e.loss_backward(A,Y)

            dA_prev, dW, db = layer.linear_activation_backward(dL_A)

            layer.recompute_weights()

            print('')

        self.layer = layer
        self.cross_e = cross_e

    def score(self, x:cp.ndarray):
        X = x.transpose()
        A, full_cache = self.layer.linear_activation_forward(X)

        A = A.transpose()

        A_label = A.copy()
        A_label[A_label > 0.5] = 1
        A_label[A_label <= 0.5] = 0

        return A_label, A


class ANN():

    layers = None
    learning_rate = None
    cost_function = None

    def __init__(self,
                 cost:str,
                 output_activation: str,
                 learning_rate:float=0.05,
                 hidden_layer_sizes: List[int] = None,
                 hidden_activations: List[str] = None
                 ):

        if hidden_layer_sizes is None:
            hidden_layer_sizes = []

        if hidden_activations is None:
            hidden_activations = []

        if cost not in ['cross-entropy']:
            raise Exception('unsupported cost function')


        for a in hidden_activations:
            if a not in ['sigmoid']:
                raise Exception('Unsupported activation function {}'.format(a))

        if output_activation not in ['sigmoid']:
            raise Exception('Unsupported activation function {}'.format(output_activation))

        if len(hidden_activations) != len(hidden_layer_sizes):
            raise Exception('hidden layer sizes and activations must be same length')


        if cost in ['cross-entropy']:
            self.cost_function = CrossEntropyCost()

        self._hidden_layer_sizes = hidden_layer_sizes
        self._hidden_activations = hidden_activations
        self.learning_rate = learning_rate
        self._output_activation = output_activation


        self.layers = []



    def _setup(self, X:cp.ndarray,y:cp.ndarray):
        '''
        Setup all the layers given the config and input/output space
        :param X: shape: [n_input_features, n_samples]
        :param y: shape: [n_outputs, n_samples]
        :return:
        '''

        if X.shape[0] > X.shape[1]:
            print(
                'WARNING: X with shape {} has more features than samples. Expected input shape is [n_input_features, n_samples]. Perhaps it needs to be transposed?'.format(X.shape))

        if y.shape[0] > y.shape[1]:
            print(
                'WARNING: y with shape {} has more outputs than samples. Expected input shape is [n_outputs, n_samples]. Perhaps it needs to be transposed?'.format(y.shape))

        self.layers = []

        #prepare the hidden layers
        for hidden_layer_number in range(len(self._hidden_layer_sizes)):

            is_starting_layer = False
            if hidden_layer_number == 0:
                is_starting_layer = True

            if is_starting_layer:
                layer_input_size = X.shape[0]
            else:
                layer_input_size = self._hidden_layer_sizes[hidden_layer_number]

            #create hidden layer
            hidden_layer = Layer(input_size=layer_input_size
                                 , layer_size=self._hidden_layer_sizes[hidden_layer_number]
                                 , activation_func=self._hidden_activations[hidden_layer_number]
                                 , learning_rate=self.learning_rate)

            self.layers.append(hidden_layer)

        #prepare the output layer
        is_starting_layer = False
        if len(self._hidden_layer_sizes) == 0:
            is_starting_layer = True

        if is_starting_layer:
            layer_input_size = X.shape[0]
        else:
            layer_input_size = self._hidden_layer_sizes[-1]

        # create output layer
        output_layer = Layer(input_size=layer_input_size
                             , layer_size=y.shape[0]
                             , activation_func=self._output_activation)

        self.layers.append(output_layer)

    def fit(self, x: cp.ndarray, y: cp.ndarray, n_epoch:int=20, cost_improvement_thresh:float=0.05):
        # shape x: [n_sample, n_feature], standard format what you get out of a df
        X = x.transpose()
        Y = y.transpose()

        #init all the layers
        self._setup(X,Y)

        last_cost = cp.inf

        for epoch in range(n_epoch):

            #forward pass
            A = X
            for layer in self.layers:
                A = layer.linear_activation_forward(A)

            current_cost = self.cost_function.compute_cost(A,Y)
            if last_cost - current_cost <= cost_improvement_thresh:
                break

            #compute derivative of loss with respect to last activation
            dL_A = self.cost_function.loss_backward(A, Y)

            #backward pass the gradient
            for layer in reversed(self.layers):
                dL_A = layer.linear_activation_backward(dL_A)

                #recompute the weights
                layer.recompute_weights()


    def score(self, x:cp.ndarray, label_cut_thresh:float=0.5):
        X = x.transpose()
        # forward pass
        A = X
        for layer in self.layers:
            A = layer.linear_activation_forward(A)

        A = A.transpose()

        A_label = A.copy()
        if label_cut_thresh is not None:
            A_label[A_label > label_cut_thresh] = 1
            A_label[A_label <= label_cut_thresh] = 0

        return A_label, A







