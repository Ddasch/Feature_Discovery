
import numpy as np
import cupy as cp
import pandas as pd
import pytest

from featurediscovery import kernel_search
from featurediscovery.util.exceptions import *
#from featurediscovery.plotter import plot_kernel

def test_to_screen_individual():
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


    kernel_search.evaluate_kernels(df
                                   , feature_space=['x1', 'x2']
                                   , target_variable='y'
                                   , monovariate_kernels=['quadratic']
                                   , eval_method='full', use_cupy='yes'
                                   , plot_ranking=False
                                   , plot_individual_kernels=True
                                   )

