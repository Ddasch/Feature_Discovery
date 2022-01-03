
import numpy as np
import cupy as cp
import pandas as pd
import pytest

from featurediscovery import kernel_search
from featurediscovery.util.exceptions import *
from featurediscovery.plotter import plot_kernel

def test_plotter_basic():
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
                                       feature_standardizers = ['raw', 'centralized', 'standard'],
                                       eval_method='full', use_cupy='yes')


    plot_kernel(df, results_cp[0], target_variable='y')



def test_scree_plot_boundary3D():

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
                                   , eval_method='full'
                                   , use_cupy='no'
                                   , plot_feature_ranking=False
                                   , plot_individual_kernels=True
                                   , kernel_plot_mode='scree'
                                   )


def test_scree_plot_boundary2D():
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

    y = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    df = pd.DataFrame(data={
        'x1': x[:, 0],
        'x2': x[:, 1],
        'y': y
    })


    kernel_search.evaluate_kernels(df
                                   , feature_space=['x1', 'x2']
                                   , target_variable='y'
                                   , monovariate_kernels=['quadratic']
                                   #, duovariate_kernels=['poly2']
                                   , feature_standardizers=['raw', 'centralized', 'standard']
                                   , eval_method='naive'
                                   , use_cupy='no'
                                   , plot_feature_ranking=False
                                   , plot_individual_kernels=True
                                   , kernel_plot_mode='scree'
                                   )



def test_ranking_to_screen():

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
                                   , monovariate_kernels=['quadratic']
                                   , duovariate_kernels=['poly2']
                                   , feature_standardizers=['raw', 'centralized', 'standard']
                                   , eval_method='full'
                                   , use_cupy='no'
                                   , plot_feature_ranking=True
                                   , plot_individual_kernels=False
                                   , kernel_plot_mode='scree'
                                   , export_folder='./test_output/'
                                   , export_ranking=True
                                   )