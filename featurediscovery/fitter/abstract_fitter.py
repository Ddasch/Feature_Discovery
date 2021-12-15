
from abc import ABC, abstractmethod
from typing import Union

import cupy as cp
import numpy as np

from featurediscovery.fitter.fit_metrics import *
from featurediscovery.fitter.abstract_fit_quality import Fit_Quality_Metric

class Abstract_Fitter(ABC):
    fit_metric:Fit_Quality_Metric=None

    def __init__(self, fit_metric:str):

        if fit_metric not in ['IG_Gini']:
            raise Exception('unsupported metric {}'.format(fit_metric))

        if fit_metric == 'IG_Gini':
            self.fit_metric = IG_Gini()



    def compute_fit_quality(self, x: Union[np.ndarray, cp.ndarray], y:Union[np.ndarray, cp.ndarray]):

        if len(x.shape) != 2:
            raise Exception('x must be 2 dimensional, first dimension indicating the sample index and second the feature index')

        if len(y.shape) != 2:
            raise Exception('y must be 2 dimensional, first dimension indicating the sample index and second the feature index')

        if y.shape[1] != 1:
            raise Exception(
                'y may only contain a single feature')

        use_cupy = type(x) == cp.ndarray

        self._fit(x,y)

        y_hat = self._score(x)

        if use_cupy:
            if len(set(cp.unique(y_hat).get()).difference(set(cp.unique(y).get()))) > 0:
                raise Exception(
                    'Fitter is returning labels not in the original set. Original set is {} but fitter is returning {}'.format(
                        set(cp.unique(y)), set(cp.unique(y_hat))))
        else:
            if len(set(np.unique(y_hat)).difference(set(np.unique(y)))) > 0:
                raise Exception(
                    'Fitter is returning labels not in the original set. Original set is {} but fitter is returning {}'.format(
                        set(np.unique(y)), set(np.unique(y_hat))))

        fit_quality = self.fit_metric.score_fit_quality(y, y_hat)

        return fit_quality


    @abstractmethod
    def _fit(self, x: Union[np.ndarray, cp.ndarray], y:Union[np.ndarray, cp.ndarray]):
        pass

    @abstractmethod
    def _score(self, x: Union[np.ndarray, cp.ndarray]):
        pass