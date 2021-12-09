
from abc import ABC, abstractmethod
from typing import Union

import cupy as cp
import numpy as np


class Abstract_Fitter(ABC):
    fit_metric=None

    def __init__(self, fit_metric:str):

        self.fit_metric = fit_metric


    def fit(self, x: Union[np.ndarray, cp.ndarray], y:Union[np.ndarray, cp.ndarray]):

        if len(x.shape) != 2:
            raise Exception('x must be 2 dimensional, first dimension indicating the sample index and second the feature index')

        if len(y.shape) != 2:
            raise Exception('y must be 2 dimensional, first dimension indicating the sample index and second the feature index')

        if y.shape[1] != 1:
            raise Exception(
                'y may only contain a single feature')

        self._fit(x,y)

        y_hat = self._score(x)

        


    @abstractmethod
    def _fit(self, x: Union[np.ndarray, cp.ndarray], y:Union[np.ndarray, cp.ndarray]):
        pass

    @abstractmethod
    def _score(self, x: Union[np.ndarray, cp.ndarray]):
        pass