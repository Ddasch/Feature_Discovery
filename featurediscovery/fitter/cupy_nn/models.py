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
    better_weight_init_method:str = None

    def __init__(self,
                 cost:str,
                 output_activation: str,
                 learning_rate:float=0.05,
                 hidden_layer_sizes: List[int] = None,
                 hidden_activations: List[str] = None,
                 better_weight_init_method:str = None,
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

        if better_weight_init_method is not None and better_weight_init_method not in ['corr', 'magnified_corr', 'corr_zero_bias', 'magnified_corr_zero_bias']:
            raise Exception('Unsupported output layer guess method')

        if cost in ['cross-entropy']:
            self.cost_function = CrossEntropyCost()

        self._hidden_layer_sizes = hidden_layer_sizes
        self._hidden_activations = hidden_activations
        self.learning_rate = learning_rate
        self._output_activation = output_activation
        self.better_weight_init_method = better_weight_init_method


        self.layers = []



    def _setup(self, X:cp.ndarray, Y:cp.ndarray, X_transposed:cp.ndarray, Y_transposed:cp.ndarray):
        '''
        Setup all the layers given the config and input/output space
        :param X: shape: [n_input_features, n_samples]
        :param Y: shape: [n_outputs, n_samples]
        :param X_transposed: shape: [n_samples, n_input_features]
        :param Y_transposed: shape: [n_samples, n_outputs]
        :return:
        '''

        if X.shape[0] > X.shape[1]:
            print(
                'WARNING: X with shape {} has more features than samples. Expected input shape is [n_input_features, n_samples]. Perhaps it needs to be transposed?'.format(X.shape))

        if Y.shape[0] > Y.shape[1]:
            print(
                'WARNING: y with shape {} has more outputs than samples. Expected input shape is [n_outputs, n_samples]. Perhaps it needs to be transposed?'.format(Y.shape))

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
                                 , learning_rate=self.learning_rate
                                 , optimizer='adam')

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
                             , layer_size=Y.shape[0]
                             , activation_func=self._output_activation
                             , learning_rate=self.learning_rate
                             , optimizer='adam')

        if len(self._hidden_layer_sizes) == 0 and self.better_weight_init_method is not None:
            #re-initialize weights with better guesses
            output_layer.init_weights_with_data(X_transposed=X_transposed
                                                , Y_transposed=Y_transposed
                                                , method=self.better_weight_init_method)

        self.layers.append(output_layer)

    def fit(self, x: cp.ndarray, y: cp.ndarray, n_epoch:int=20
            , cost_improvement_thresh:float=0.001
            , cost_improvement_agg_range:int=5
            , verbose:bool=False):

        if n_epoch is None:
            n_epoch = 99999999

        # shape x: [n_sample, n_feature], standard format what you get out of a df
        X = x.transpose()
        Y = y.transpose()

        #init all the layers
        self._setup(X,Y, x,y)

        #last_cost = cp.inf

        cost_history = cp.ones((cost_improvement_agg_range))*99999

        for epoch in range(n_epoch):

            #forward pass
            A = X
            for layer in self.layers:
                A = layer.linear_activation_forward(A)

            #check cost criteria for early stopping
            current_cost = self.cost_function.compute_cost(A,Y)
            cost_history_agg = cp.median(cost_history)

            if verbose:
                print('Cost at epoch {} : {}'.format(epoch, current_cost))

            if cost_improvement_thresh is not None:
                if cost_history_agg - current_cost <= cost_improvement_thresh:
                    break

            cost_history[:-1] = cost_history[1:]
            cost_history[-1] = current_cost

            #compute derivative of loss with respect to last activation
            dL_A = self.cost_function.loss_backward(A, Y)

            #backward pass the gradient
            for layer in reversed(self.layers):
                dL_A = layer.linear_activation_backward(dL_A)

                #recompute the weights
                layer.recompute_weights(iteration=epoch+1)


        self._n_performed_epochs = epoch


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







