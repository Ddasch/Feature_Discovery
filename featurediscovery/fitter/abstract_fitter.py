
from abc import ABC, abstractmethod
from typing import Union

import cupy as cp
import numpy as np

from featurediscovery.fitter.fit_metrics import *
from featurediscovery.fitter.abstract_fit_quality import Fit_Quality_Metric

class Abstract_Fitter(ABC):
    fit_metric:Fit_Quality_Metric=None

    def __init__(self, fit_metric:str):

        if fit_metric not in ['IG_Gini', 'acc']:
            raise Exception('unsupported metric {}'.format(fit_metric))

        if fit_metric == 'IG_Gini':
            self.fit_metric = IG_Gini()

        if fit_metric == 'acc':
            self.fit_metric = Accuracy()



    def compute_fit_quality(self, x: Union[np.ndarray, cp.ndarray], y:Union[np.ndarray, cp.ndarray]) -> float:

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
                y_hat = self._score(x)
                raise Exception(
                    'Fitter is returning labels not in the original set. Original set is {} but fitter is returning {}'.format(
                        set(cp.unique(y).get()), set(cp.unique(y_hat).get())))
        else:
            if len(set(np.unique(y_hat)).difference(set(np.unique(y)))) > 0:
                raise Exception(
                    'Fitter is returning labels not in the original set. Original set is {} but fitter is returning {}'.format(
                        set(np.unique(y)), set(np.unique(y_hat))))

        fit_quality = self.fit_metric.score_fit_quality(y, y_hat)

        return fit_quality


    def compute_decision_boundary_samples(self, x: Union[np.ndarray, cp.ndarray]):
        # perform a grid-score and extract all the points which are near the decision boundary

        #use_cupy = type(x) == cp.ndarray
        api = np
        if type(x) == cp.ndarray:
            api = cp

        x_mins = x.min(axis=0)
        x_maxs = x.max(axis=0)
        n_dims = len(x_mins)

        n_boundary_samples = 5
        min_decision_samples = 100
        # tolerance range of [0.5-tolerance, 0.5+tolerance] of scores considered to be close enough to decision boundary
        y_prob_tolerance = 0.01

        enough_decision_boundary_points = False

        while not enough_decision_boundary_points:
            linspaces = []
            '''
            if use_cupy:
                for d_index in range(n_dims):
                    linspaces.append(cp.linspace(x_mins[d_index], x_maxs[d_index], n_boundary_samples))
                x_linspace = cp.array(cp.meshgrid(*linspaces)).T.reshape(-1, n_dims)
    
            if not use_cupy:
                for d_index in range(n_dims):
                    linspaces.append(np.linspace(x_mins[d_index], x_maxs[d_index], n_boundary_samples))
                x_linspace = np.array(np.meshgrid(*linspaces)).T.reshape(-1, n_dims)
            '''
            for d_index in range(n_dims):
                linspaces.append(api.linspace(x_mins[d_index], x_maxs[d_index], n_boundary_samples))
            x_linspace = api.array(api.meshgrid(*linspaces)).T.reshape(-1, n_dims)

            y_hat_probs = self._score_prob(x_linspace)

            y_hat_decision_mask = (y_hat_probs > 0.5 - y_prob_tolerance) & (y_hat_probs < 0.5 + y_prob_tolerance)
            y_hat_decision_mask_samples = [x for x in range(x_linspace.shape[0]) if y_hat_decision_mask[x][0]]
            y_hat_decision_boundary = y_hat_probs[y_hat_decision_mask]

            amount_of_samples_within_boundary = len(y_hat_decision_boundary)
            if amount_of_samples_within_boundary >= min_decision_samples:
                enough_decision_boundary_points = True
            else:
                if len(y_hat_probs[y_hat_probs < 0.5].reshape(-1)) in [0, y_hat_probs.shape[0]]:
                    #special abort procedure, model predicts the same for entire input space
                    enough_decision_boundary_points = True

                n_boundary_samples = int(n_boundary_samples * 1.5)

            # if use_cupy:
            #    y_hat_decision_mask_2d = cp.ones((n_dims), dtype=bool) * y_hat_decision_mask
            # else:
            #    y_hat_decision_mask_2d = np.ones((n_dims), dtype=bool) * y_hat_decision_mask

        x_decision_boundary = x_linspace[y_hat_decision_mask_samples, :].copy()

        return x_decision_boundary, y_hat_decision_boundary

    @abstractmethod
    def _fit(self, x: Union[np.ndarray, cp.ndarray], y:Union[np.ndarray, cp.ndarray]):
        pass

    @abstractmethod
    def _score(self, x: Union[np.ndarray, cp.ndarray]):
        pass

    @abstractmethod
    def _score_prob(self, x: Union[np.ndarray, cp.ndarray]):
        pass
