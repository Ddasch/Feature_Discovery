
from typing import Union, List

import cupy as cp
import numpy as np
from sklearn.metrics import accuracy_score

from featurediscovery.fitter.abstract_fit_quality import Fit_Quality_Metric


class IG_Gini(Fit_Quality_Metric):

    def score_fit_quality(self, y: Union[np.ndarray, cp.ndarray]
                          , y_hat: Union[np.ndarray, cp.ndarray]):
        use_cupy = type(y) == cp.ndarray

        pre_fit_gini = gini(y)
        post_fit_ginis = []
        post_fit_sample_sizes = []

        if use_cupy:
            uniques = cp.unique(y)
        else:
            uniques = np.unique(y)

        for c in uniques:
            mask = y_hat == c
            y_masked_selection = y[mask]
            post_fit_ginis.append(gini(y_masked_selection))
            post_fit_sample_sizes.append(float(len(y_masked_selection)))

        if use_cupy:
            post_fit_ginis = cp.array(post_fit_ginis)
            post_fit_sample_sizes = cp.array(post_fit_sample_sizes)
            weighted_average_post_fit_gini = cp.average(post_fit_ginis, weights=post_fit_sample_sizes / cp.sum(post_fit_sample_sizes))

            weighted_average_post_fit_gini = weighted_average_post_fit_gini.get()
            pre_fit_gini = pre_fit_gini
        else:
            post_fit_ginis = np.array(post_fit_ginis)
            post_fit_sample_sizes = np.array(post_fit_sample_sizes)
            weighted_average_post_fit_gini = np.average(post_fit_ginis,
                                                        weights=post_fit_sample_sizes / np.sum(post_fit_sample_sizes))

        return pre_fit_gini - weighted_average_post_fit_gini



class Accuracy(Fit_Quality_Metric):

    def score_fit_quality(self, y: Union[np.ndarray, cp.ndarray]
                          , y_hat: Union[np.ndarray, cp.ndarray]):
        use_cupy = type(y) == cp.ndarray

        if use_cupy:
            y = y.get()
            y_hat = y_hat.get()

        return accuracy_score(y,y_hat)



def gini(y: Union[np.ndarray, cp.ndarray]) -> float:
    use_cupy = type(y) == cp.ndarray

    impurity = 0.0

    n_total = float(len(y))

    if use_cupy:
        uniques = cp.unique(y)
    else:
        uniques = np.unique(y)

    for c in uniques:
        prob_c = len(y[y == c]) / n_total
        prob_not_c = 1 - prob_c
        impurity = impurity + (prob_c * prob_not_c)

    return impurity
