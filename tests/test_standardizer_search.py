
import numpy as np
import cupy as cp
import pandas as pd
import pytest
import os
import shutil

from featurediscovery import kernel_search
from featurediscovery.util.exceptions import *




def test_all_standardizers():

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


    kernel_search.evaluate_kernels(df
                                   , feature_space=['x1', 'x2', 'x3']
                                   , target_variable='y'
                                   #, monovariate_kernels=['quadratic']
                                   , duovariate_kernels=['poly2']
                                   , feature_standardizers=['raw', 'standard', 'centralized']
                                   , eval_method='full'
                                   , use_cupy='no'
                                   , plot_ranking=True
                                   , plot_individual_kernels=False
                                   , kernel_plot_mode='scree'
                                   , export_folder='./test_output/'
                                   , export_formats=['json']
                                   , export_ranking=False
                                   )