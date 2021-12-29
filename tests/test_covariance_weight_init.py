
import numpy as np
import cupy as cp
import pandas as pd
import pytest

from featurediscovery.fitter.cupy_nn.weight_initializer import _cross_covariance, _cross_corr
from featurediscovery.fitter.cupy_nn.models import ANN

def test_covariance_comp():
    x = np.array([
        np.array([-3, 0, 9]),
        np.array([-2, 1, 4]),
        np.array([-1, 0, 1]),
        np.array([0, 1, 0]),
        np.array([1, 0, 1]),
        np.array([2, 1, 4]),
        np.array([3, 0, 9]),
        np.array([-4, 1, 16]),
        np.array([4, 0, 16]),
        np.array([-5, 1, 25]),
        np.array([5, 0, 25]),
        np.array([-6, 1, 36]),
        np.array([6, 0, 36])
    ])

    y = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,1]).reshape(-1,1)


    X = cp.array(x)
    Y = cp.array(y)

    W = _cross_covariance(X, Y)

    assert W.shape[0] == 3
    assert W.shape[1] == 1


def test_covariance_comp_scale():
    x = np.array([
        np.array([-3, 0, 9]),
        np.array([-2, 1, 4]),
        np.array([-1, 0, 1]),
        np.array([0, 1, 0]),
        np.array([1, 0, 1]),
        np.array([2, 1, 4]),
        np.array([3, 0, 9]),
        np.array([-4, 1, 16]),
        np.array([4, 0, 16]),
        np.array([-5, 1, 25]),
        np.array([5, 0, 25]),
        np.array([-6, 1, 36]),
        np.array([6, 0, 36])
    ])

    #y = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]).reshape(-1,1)

    y = np.array([
        np.array([0, 1]),
        np.array([0, 1]),
        np.array([0, 1]),
        np.array([0, 1]),
        np.array([0, 1]),
        np.array([0, 1]),
        np.array([0, 1]),
        np.array([1, 0]),
        np.array([1, 0]),
        np.array([1, 0]),
        np.array([1, 0]),
        np.array([1, 0]),
        np.array([1, 0])
    ])

    X = cp.array(x)
    Y = cp.array(y)

    W = _cross_covariance(X, Y)

    assert W.shape[0] == 3
    assert W.shape[1] == 2



def test_corr_comp_scale():
    x = np.array([
        np.array([-3, 0, 9]),
        np.array([-2, 1, 4]),
        np.array([-1, 0, 1]),
        np.array([0, 1, 0]),
        np.array([1, 0, 1]),
        np.array([2, 1, 4]),
        np.array([3, 0, 9]),
        np.array([-4, 1, 16]),
        np.array([4, 0, 16]),
        np.array([-5, 1, 25]),
        np.array([5, 0, 25]),
        np.array([-6, 1, 36]),
        np.array([6, 0, 36])
    ])

    #y = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]).reshape(-1,1)

    y = np.array([
        np.array([0, 2]),
        np.array([0, 2]),
        np.array([0, 2]),
        np.array([0, 2]),
        np.array([0, 2]),
        np.array([0, 2]),
        np.array([0, 2]),
        np.array([1, -1]),
        np.array([1, -1]),
        np.array([1, -1]),
        np.array([1, -1]),
        np.array([1, -1]),
        np.array([1, -1])
    ])

    X = cp.array(x)
    Y = cp.array(y)

    W = _cross_corr(X, Y)

    assert W.shape[0] == 3
    assert W.shape[1] == 2


def test_weight_init_fit():
    x = np.array([
        np.array([-3, 0, 9]),
        np.array([-2, 1, 4]),
        np.array([-1, 0, 1]),
        np.array([0, 1, 0]),
        np.array([1, 0, 1]),
        np.array([2, 1, 4]),
        np.array([3, 0, 9]),
        np.array([-4, 1, 16]),
        np.array([4, 0, 16]),
        np.array([-5, 1, 25]),
        np.array([5, 0, 25]),
        np.array([-6, 1, 36]),
        np.array([6, 0, 36])
    ])

    y = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]).reshape(-1, 1)

    X = cp.array(x)
    Y = cp.array(y)


    n_epochs_better = []
    n_epochs_random = []

    for i in range(100):
        model =  ANN(cost='cross-entropy'
                     , output_activation='sigmoid'
                     , hidden_activations=None
                     , hidden_layer_sizes=None
                     , learning_rate=0.1
                     , better_weight_init_method='corr')

        model.fit(X, Y, verbose=False, max_epoch=200)
        n_epochs_better.append(model._n_performed_epochs)
        y_hat = model.score(X)

    for i in range(100):
        model2 = ANN(cost='cross-entropy'
                     , output_activation='sigmoid'
                     , hidden_activations=None
                     , hidden_layer_sizes=None
                     , learning_rate=0.1
                     , better_weight_init_method=None)

        model2.fit(X, Y, verbose=False, max_epoch=200)
        n_epochs_random.append(model2._n_performed_epochs)
        y_hat2 = model2.score(X)


    mean_epochs_better = np.mean(n_epochs_better)
    mean_epochs_random = np.mean(n_epochs_random)

    print('AVG epochs with better guesser: {}'.format(mean_epochs_better))
    print('AVG epochs with glorot: {}'.format(mean_epochs_random))






