from itertools import combinations
from abc import ABC, abstractmethod
from typing import Union, List

import cupy as cp
import numpy as np

from featurediscovery.fitter.abstract_fit_quality import Fit_Quality_Metric


class Gini_Metric(Fit_Quality_Metric):

    def score_fit_improvement(self, y: Union[np.ndarray, cp.ndarray]
                              , y_hat: Union[np.ndarray, cp.ndarray]):

        pre_fit_gini = gini(y)

        post_fit_ginis = []
        post_fit_sample_sizes = []

        for c in cp.unique(y):
            mask = y_hat == c
            y_masked_selection = y[mask]
            post_fit_ginis.append(gini(y_masked_selection))
            post_fit_sample_sizes.append(float(len(y_masked_selection)))


        post_fit_ginis = cp.array(post_fit_ginis)
        post_fit_sample_sizes = cp.array(post_fit_sample_sizes)

        weighted_average_post_fit_gini = cp.average(post_fit_ginis, weights=post_fit_sample_sizes / cp.sum(post_fit_sample_sizes))

        return pre_fit_gini - weighted_average_post_fit_gini



def gini(y: Union[np.ndarray, cp.ndarray]) -> float:
    impurity = 0.0

    n_total = float(len(y))
    for c in cp.unique(y):
        prob_c = len(y[y == c]) / n_total
        prob_not_c = 1 - prob_c
        impurity = impurity + (prob_c * prob_not_c)

    return impurity
