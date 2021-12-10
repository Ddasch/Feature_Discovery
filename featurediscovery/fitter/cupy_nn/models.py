import cupy as cp

from fitter.cupy_nn.costs import CrossEntropyCost
from fitter.cupy_nn.layers import Layer


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
        A_label[A_label > 0.5] = 1
        A_label[A_label <= 0.5] = 0

        return A_label, A