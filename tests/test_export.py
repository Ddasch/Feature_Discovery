
import numpy as np
import cupy as cp
import pandas as pd
import pytest
import os
import shutil

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
                                   , feature_standardizers=['raw']
                                   , eval_method='full'
                                   , use_cupy='yes'
                                   , plot_ranking=False
                                   , plot_individual_kernels=True
                                   )



def test_to_file_individual():
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

    export_folder = './test_output/'

    if os.path.exists(export_folder + 'figures'):
        shutil.rmtree(export_folder + 'figures')

    assert not os.path.isdir(export_folder + 'figures')

    kernel_search.evaluate_kernels(df
                                   , feature_space=['x1', 'x2']
                                   , target_variable='y'
                                   , monovariate_kernels=['quadratic']
                                   , feature_standardizers=['raw']
                                   , eval_method='full'
                                   , use_cupy='yes'
                                   , plot_ranking=False
                                   , plot_individual_kernels=False
                                   , export_folder=export_folder
                                   , export_individual_kernel_plots=True
                                   )

    assert os.path.isdir(export_folder + 'figures')




def test_to_screen_individual_tsne():

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
                                   , feature_standardizers=['raw']
                                   , eval_method='full'
                                   , use_cupy='yes'
                                   , plot_ranking=False
                                   , plot_individual_kernels=True
                                   , kernel_plot_mode='tsne'
                                   )


def test_to_screen_individual_Duovariate_scree():

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
                                   , feature_space=['x1', 'x2']
                                   , target_variable='y'
                                   #, monovariate_kernels=['quadratic']
                                   , duovariate_kernels=['poly2']
                                   , feature_standardizers=['raw']
                                   , eval_method='full'
                                   , use_cupy='no'
                                   , plot_ranking=False
                                   , plot_individual_kernels=True
                                   , kernel_plot_mode='scree'
                                   )



def test_to_csv_performances():

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
                                   , plot_ranking=False
                                   , plot_individual_kernels=False
                                   , kernel_plot_mode='scree'
                                   , export_folder='./test_output/'
                                   , export_formats=['csv']
                                   , export_ranking=True
                                   )


def test_to_json_performances():

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
                                   , feature_standardizers=['raw']
                                   , eval_method='full'
                                   , use_cupy='no'
                                   , plot_ranking=False
                                   , plot_individual_kernels=False
                                   , kernel_plot_mode='scree'
                                   , export_folder='./test_output/'
                                   , export_formats=['json']
                                   , export_ranking=True
                                   )