

import numpy as np
import cupy as cp
import pandas as pd
import pytest

from featurediscovery import kernel_search
from featurediscovery.util.exceptions import *


def test_run_script_full_monovariate_cupy_apply():
    x = np.array([
        np.array([-3, 0]),
        np.array([-2, 1]),
        np.array([-1, 0]),
        np.array([0, 1]),
        np.array([1, 0]),
        np.array([2, 1]),
        np.array([3, 0]),
        np.array([-4, 1]),
        np.array([4, 0]),
        np.array([-5, 1]),
        np.array([5, 0]),
        np.array([-6, 1]),
        np.array([6, 0])
    ])

    y = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,1])

    df = pd.DataFrame(data={
        'x1': x[:, 0],
        'x2': x[:, 1],
        'y': y
    })

    results_cp = kernel_search._search(df, feature_space=['x1', 'x2'], target_variable='y',
                                       monovariate_kernels=['quadratic'],
                                       eval_method='full', use_cupy='yes')


    df_with_kernel = results_cp[0].apply(df)

    assert 'x1^2' in df_with_kernel.columns

