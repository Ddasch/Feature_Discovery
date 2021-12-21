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
                         , learning_rate=0.04
                         , better_weight_init_method='magnified_corr')

    def _fit(self, x: Union[np.ndarray, cp.ndarray], y:Union[np.ndarray, cp.ndarray]):
        self.model.fit(x,y,n_epoch=300, cost_improvement_thresh=0.0001, cost_improvement_agg_range=5, verbose=False)


    def _score(self, x: Union[np.ndarray, cp.ndarray]):
        return self.model.score(x)[0]

    def _score_prob(self, x: Union[np.ndarray, cp.ndarray]):
        return self.model.score(x)[1]

