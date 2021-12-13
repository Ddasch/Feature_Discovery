import cupy as cp
import numpy as np
from typing import Union
from featurediscovery.fitter.abstract_fitter import Abstract_Fitter

from featurediscovery.fitter.cupy_nn.models import ANN


class Logistic_Regression(Abstract_Fitter):

    model:ANN = None

    def __init__(self, fit_metric:str):
        super().__init__(fit_metric)

        self.model = ANN(cost='cross-entropy'
                         , output_activation='sigmoid'
                         , hidden_activations=None
                         , hidden_layer_sizes=None)

    def _fit(self, x: Union[np.ndarray, cp.ndarray], y:Union[np.ndarray, cp.ndarray]):
        self.model.fit(x,y,n_epoch=20)


    def _score(self, x: Union[np.ndarray, cp.ndarray]):
        return self.model.score(x)[0]




