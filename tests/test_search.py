

import numpy as np
import cupy as cp
import pandas as pd
import pytest

from featurediscovery import kernel_search
from featurediscovery.util.exceptions import *



def test_run_script_execution():
    x = np.array([
        np.array([1, 2]),
        np.array([1, 3]),
        np.array([1, 4]),
        np.array([2, 3]),
        np.array([2, 4]),
        np.array([3, 4]),
        np.array([2, 1]),
        np.array([3, 1]),
        np.array([3, 2]),
        np.array([4, 1]),
        np.array([4, 2]),
        np.array([4, 3])
    ])

    y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    df = pd.DataFrame(data={
        'x1': x[:, 0],
        'x2': x[:, 1],
        'y': y
    })


    results = kernel_search.search(df, feature_space=['x1', 'x2'], target_variable='y', monovariate_kernels=['quadratic'],
                                   eval_method='naive', use_cupy='yes')

    assert results is not None



def test_run_script_full_monovariate():
    x = np.array([
        np.array([0, 2]),
        np.array([0.2, 3]),
        np.array([1, 4]),
        np.array([1.2, 3]),
        np.array([1.5, 4]),
        np.array([2, 4]),
        np.array([1.8, 1]),
        np.array([2.5, 1]),
        np.array([3, 2]),
        np.array([0.8, 1]),
        np.array([0.5, 2]),
        np.array([2.1, 3])
    ])

    y = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

    df = pd.DataFrame(data={
        'x1': x[:, 0],
        'x2': x[:, 1],
        'y': y
    })


    results = kernel_search.search(df, feature_space=['x1', 'x2'], target_variable='y', monovariate_kernels=['quadratic'],
                                   eval_method='full', use_cupy='yes')

    assert results is not None



def test_run_script_full_monovariate_numpy():
    x = np.array([
        np.array([0, 2]),
        np.array([0.2, 3]),
        np.array([1, 4]),
        np.array([1.2, 3]),
        np.array([1.5, 4]),
        np.array([2, 4]),
        np.array([1.8, 1]),
        np.array([2.5, 1]),
        np.array([3, 2]),
        np.array([0.8, 1]),
        np.array([0.5, 2]),
        np.array([2.1, 3])
    ])

    y = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

    df = pd.DataFrame(data={
        'x1': x[:, 0],
        'x2': x[:, 1],
        'y': y
    })


    results = kernel_search.search(df, feature_space=['x1', 'x2'], target_variable='y', monovariate_kernels=['quadratic'],
                                   eval_method='full', use_cupy='no')

    assert results is not None

