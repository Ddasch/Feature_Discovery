import cupy as cp
import numpy as np
from typing import Union
from featurediscovery.fitter.abstract_fitter import Abstract_Fitter

from featurediscovery.fitter.cupy_nn.models import ANN

class Logistic_Regression_ANN(Abstract_Fitter):

    model:ANN = None

    def __init__(self, fit_metric:str):
        super().__init__(fit_metric)

        self.model = ANN(cost='cross-entropy'
                         , output_activation='sigmoid'
                         , hidden_activations=None
                         , hidden_layer_sizes=None
                         , learning_rate=0.1
                         , better_weight_init_method='magnified_corr')

    def _fit(self, x: Union[np.ndarray, cp.ndarray], y:Union[np.ndarray, cp.ndarray], debug:bool=False):
        self.model.fit(x, y
                       , max_epoch=500
                       , min_epoch=5
                       , cost_improvement_thresh=0.00001
                       , cost_improvement_agg_range=10
                       , verbose=False, debug=debug)


    def _score(self, x: Union[np.ndarray, cp.ndarray]):
        return self.model.score(x)[0]

    def _score_prob(self, x: Union[np.ndarray, cp.ndarray]):
        return self.model.score(x)[1]

