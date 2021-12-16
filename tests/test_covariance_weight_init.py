
import numpy as np
import cupy as cp
import pandas as pd
import pytest

from featurediscovery.fitter.cupy_nn.weight_initializer import covariance


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

    W = covariance(X,Y)

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

    W = covariance(X,Y)

    assert W.shape[0] == 3
    assert W.shape[1] == 2





