

import numpy as np
import cupy as cp
import pandas as pd
import pytest

from featurediscovery import kernel_search



def test_full_duovariate_numpy_apply():
    x = np.array([
        np.array([-3, 0, 1]),
        np.array([-2, 1, 2]),
        np.array([-1, 0, 3]),
        np.array([0, 1, 1]),
        np.array([1, 0, 2]),
        np.array([2, 1, 3]),
        np.array([3, 0, 1]),
        np.array([-4, 1, 2]),
        np.array([4, 0, 3]),
        np.array([-5, 1, 1]),
        np.array([5, 0, 2]),
        np.array([-6, 1, 3]),
        np.array([6, 0, 1])
    ])

    y = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    df = pd.DataFrame(data={
        'x1': x[:, 0],
        'x2': x[:, 1],
        'x3': x[:, 2],
        'y': y
    })


    results = kernel_search._search(df, feature_space=['x1', 'x2', 'x3'], target_variable='y',
                                    duovariate_kernels=['poly2'],
                                    eval_method='full', use_cupy='no')

    df_kernel = results[0].apply(df)

    assert len(df.columns) == 7