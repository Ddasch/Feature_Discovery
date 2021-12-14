
import numpy as np
import cupy as cp
import pandas as pd
import pytest

from featurediscovery import kernel_search
from featurediscovery.util.exceptions import *

def test_kernel_init():
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
        'x1': x[:,0],
        'x2': x[:,1],
        'y': y
    })

    df_wrong = pd.DataFrame(data={
        'x1': x[:,0],
        'x3': x[:,1],
        'y': y
    })


    with pytest.raises(TypeError) as e:
        kernel_search.search()

    with pytest.raises(TypeError) as e:
        kernel_search.search(df)

    with pytest.raises(TypeError) as e:
        kernel_search.search(feature_space=['x1', 'x2'], target_variable='y')

    with pytest.raises(TypeError) as e:
        kernel_search.search(df, feature_space=['x1', 'x2'])

    with pytest.raises(SetupException) as e:
        kernel_search.search(df, feature_space=['x1', 'x2'], target_variable='y')

    with pytest.raises(SetupException) as e:
        kernel_search.search(df, feature_space=['x1', 'x3'], target_variable='y', monovariate_kernels=['quadratic'])

    with pytest.raises(SetupException) as e:
        kernel_search.search(df, feature_space=['x1', 'x2'], target_variable='y1', monovariate_kernels=['quadratic'])

    with pytest.raises(SetupException) as e:
        kernel_search.search(df, feature_space=['x1', 'x2'], target_variable=['y'], monovariate_kernels=['quadratic'])

    with pytest.raises(SetupException) as e:
        kernel_search.search(df, feature_space='x1,x2', target_variable='y', monovariate_kernels=['quadratic'])

    with pytest.raises(SetupException) as e:
        kernel_search.search(df_wrong, feature_space=['x1', 'x2'], target_variable='y', monovariate_kernels=['quadratic'])

    with pytest.raises(SetupException) as e:
        kernel_search.search(df, feature_space=['x1', 'x2'], target_variable='y', monovariate_kernels='quadratic')

    with pytest.raises(SetupException) as e:
        kernel_search.search(df, feature_space=['x1', 'x2'], target_variable='y', monovariate_kernels=['magic'])

    with pytest.raises(SetupException) as e:
        kernel_search.search(df, feature_space=['x1', 'x2'], target_variable='y', monovariate_kernels=[], duovariate_kernels=[])

    with pytest.raises(SetupException) as e:
        kernel_search.search(df, feature_space=['x1', 1], target_variable='y', monovariate_kernels=['quadratic'])

    print('')
