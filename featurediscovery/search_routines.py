

import numpy as np
import cupy as cp
import pandas as pd
from typing import List

from featurediscovery.kernels.abstract_kernel import Abstract_Kernel
from featurediscovery.fitter.cupy_fitter import Logistic_Regression_ANN
from featurediscovery.fitter.sklearn_fitters import Logistic_Scikit
from featurediscovery.kernels.monovariate.monovariate_kernels import get_mono_kernel
from featurediscovery.kernels.duovariate.duovariate_kernels import get_duo_kernel


def naive_monovariate(df:pd.DataFrame
                      , search_dicts:List[dict]
                      , target_variable:str
                      , feature_space:List[str]
                      , compute_decision_boundary:bool=False
                      , use_cupy:bool=True) -> List[Abstract_Kernel]:

    #if use_cupy:
    #    return _naive_monovariate_cupy(df, search_dicts, target_variable=target_variable, feature_space=feature_space)
    #else:
    #    return _naive_monovariate_numpy(df, search_dicts, target_variable=target_variable, feature_space=feature_space)
    return _generic(df, search_dicts, target_variable=target_variable, feature_space=feature_space,
                    use_cupy=use_cupy, search_method='naive', kernel_type='monovariate',
                    compute_decision_boundary=compute_decision_boundary)


def full_monovariate(df:pd.DataFrame
                      , search_dicts:List[dict]
                      , target_variable:str
                      , feature_space:List[str]
                      , compute_decision_boundary:bool=False
                      , use_cupy:bool=True) -> List[Abstract_Kernel]:

    #if use_cupy:
    #    return _full_monovariate_cupy(df, search_dicts, target_variable=target_variable, feature_space=feature_space)
    #else:
    #    return _full_monovariate_numpy(df, search_dicts, target_variable=target_variable, feature_space=feature_space)
    return _generic(df, search_dicts, target_variable=target_variable, feature_space=feature_space,
                    use_cupy=use_cupy, search_method='full', kernel_type='monovariate',
                    compute_decision_boundary=compute_decision_boundary)

def naive_duovariate(df:pd.DataFrame
                      , search_dicts:List[dict]
                      , target_variable:str
                      , feature_space:List[str]
                      , compute_decision_boundary: bool = False
                      , use_cupy:bool=True) -> List[Abstract_Kernel]:
    #if use_cupy:
    #    return _naive_duovariate_cupy(df, search_dicts, target_variable=target_variable, feature_space=feature_space)
    #else:
    #    return _naive_duovariate_numpy(df, search_dicts, target_variable=target_variable, feature_space=feature_space)
    return _generic(df, search_dicts, target_variable=target_variable, feature_space=feature_space,
                    use_cupy=use_cupy, search_method='naive', kernel_type='duovariate',
                    compute_decision_boundary=compute_decision_boundary)


def full_duovariate(df:pd.DataFrame
                      , search_dicts:List[dict]
                      , target_variable:str
                      , feature_space:List[str]
                      , compute_decision_boundary: bool = False
                      , use_cupy:bool=True) -> List[Abstract_Kernel]:
    #if use_cupy:
    #    return _full_duovariate_cupy(df, search_dicts, target_variable=target_variable, feature_space=feature_space)
    #else:
    #    return _full_duovariate_numpy(df, search_dicts, target_variable=target_variable, feature_space=feature_space)

    return _generic(df, search_dicts, target_variable=target_variable, feature_space=feature_space,
                    use_cupy=use_cupy, search_method='full', kernel_type='duovariate',
                    compute_decision_boundary=compute_decision_boundary)


'''
#################
## MONOVARIATE ##
#################

def _naive_monovariate_cupy(df:pd.DataFrame
                      , search_dicts:List[dict]
                      , target_variable:str
                      , feature_space:List[str]
                      , quality_metric:str='IG_Gini') -> List[Abstract_Kernel]:


    X = df[feature_space].to_numpy(dtype=np.float64)
    Y = df[target_variable].to_numpy(dtype=np.float64).reshape((-1,1))

    X = cp.array(X)
    Y = cp.array(Y)

    feature_name_2_index = {}
    for i in range(len(feature_space)):
        feature_name_2_index[feature_space[i]] = i

    all_kernels = []

    for kernel_dict in search_dicts:
        #slice X so that it only has the feature for the input kernel
        feature_index = feature_name_2_index[kernel_dict['feature_a']]
        X_slice = X[:,[feature_index]]

        #apply the kernel function to the selected feature
        kernel = get_mono_kernel(kernel_dict['kernel'])
        X_kernel = kernel.fit_and_transform(X_slice)

        #evaluate naive fit quality
        fitter = Logistic_Regression_ANN(quality_metric)
        fit_quality = fitter.compute_fit_quality(X_kernel,Y)

        # estimate decision boundary
        x_decision_boundary, y_decision_boundary = fitter.compute_decision_boundary_samples(X_kernel)

        # finalize result in kernel
        kernel.finalize(fit_quality, [kernel_dict['feature_a']]
                        , x_decision_boundary, y_decision_boundary)

        #append to list
        all_kernels.append(kernel)

    return all_kernels





def _naive_monovariate_numpy(df:pd.DataFrame
                      , search_dicts:List[dict]
                      , target_variable:str
                      , feature_space:List[str]
                      , quality_metric: str = 'IG_Gini') -> List[Abstract_Kernel]:
    X = df[feature_space].to_numpy(dtype=np.float64)
    Y = df[target_variable].to_numpy(dtype=np.float64).reshape((-1, 1))


    feature_name_2_index = {}
    for i in range(len(feature_space)):
        feature_name_2_index[feature_space[i]] = i

    all_kernels = []

    for kernel_dict in search_dicts:
        # slice X so that it only has the feature for the input kernel
        feature_index = feature_name_2_index[kernel_dict['feature_a']]
        X_slice = X[:, [feature_index]]

        # apply the kernel function to the selected feature
        kernel = get_mono_kernel(kernel_dict['kernel'])
        X_kernel = kernel.fit_and_transform(X_slice)

        # evaluate naive fit quality
        fitter = Logistic_Scikit(quality_metric)
        fit_quality = fitter.compute_fit_quality(X_kernel, Y)

        # estimate decision boundary
        x_decision_boundary, y_decision_boundary = fitter.compute_decision_boundary_samples(X_kernel)

        # finalize result in kernel
        kernel.finalize(fit_quality, [kernel_dict['feature_a']]
                        , x_decision_boundary, y_decision_boundary)

        # append to list
        all_kernels.append(kernel)

    return all_kernels


def _full_monovariate_cupy(df:pd.DataFrame
                      , search_dicts:List[dict]
                      , target_variable:str
                      , feature_space:List[str]
                      , quality_metric: str = 'IG_Gini') -> List[Abstract_Kernel]:
    X = df[feature_space].to_numpy(dtype=np.float64)
    Y = df[target_variable].to_numpy(dtype=np.float64).reshape((-1,1))

    X = cp.array(X)
    Y = cp.array(Y)

    feature_name_2_index = {}
    for i in range(len(feature_space)):
        feature_name_2_index[feature_space[i]] = i

    all_kernels = []

    #compute separability prior kernel extension
    fitter = Logistic_Regression_ANN(quality_metric)
    fit_quality_pre_kernel = fitter.compute_fit_quality(X, Y)

    for kernel_dict in search_dicts:
        #slice X so that it only has the feature for the input kernel
        feature_index = feature_name_2_index[kernel_dict['feature_a']]
        X_slice = X[:,[feature_index]]

        #apply the kernel function to the selected feature
        kernel = get_mono_kernel(kernel_dict['kernel'])
        X_kernel = kernel.fit_and_transform(X_slice)

        X_stacked = cp.column_stack((X, X_kernel))

        #evaluate full fit quality
        fitter = Logistic_Regression_ANN(quality_metric)
        fit_quality = fitter.compute_fit_quality(X_stacked,Y)

        fit_improvement = fit_quality - fit_quality_pre_kernel

        # estimate decision boundary
        x_decision_boundary, y_decision_boundary = fitter.compute_decision_boundary_samples(X_stacked)

        # finalize result in kernel
        kernel.finalize(fit_improvement, [kernel_dict['feature_a']]
                        , x_decision_boundary, y_decision_boundary)

        #append to list
        all_kernels.append(kernel)

    return all_kernels



def _full_monovariate_numpy(df:pd.DataFrame
                      , search_dicts:List[dict]
                      , target_variable:str
                      , feature_space:List[str]
                      , quality_metric: str = 'IG_Gini') -> List[Abstract_Kernel]:
    X = df[feature_space].to_numpy(dtype=np.float64)
    Y = df[target_variable].to_numpy(dtype=np.float64).reshape((-1,1))


    feature_name_2_index = {}
    for i in range(len(feature_space)):
        feature_name_2_index[feature_space[i]] = i

    all_kernels = []

    #compute separability prior kernel extension
    fitter = Logistic_Scikit(quality_metric)
    fit_quality_pre_kernel = fitter.compute_fit_quality(X, Y)

    for kernel_dict in search_dicts:
        #slice X so that it only has the feature for the input kernel
        feature_index = feature_name_2_index[kernel_dict['feature_a']]
        X_slice = X[:,[feature_index]]

        #apply the kernel function to the selected feature
        kernel = get_mono_kernel(kernel_dict['kernel'])
        X_kernel = kernel.fit_and_transform(X_slice)

        X_stacked = np.column_stack((X, X_kernel))

        #evaluate full fit quality
        fitter = Logistic_Scikit(quality_metric)
        fit_quality = fitter.compute_fit_quality(X_stacked,Y)

        fit_improvement = fit_quality - fit_quality_pre_kernel

        # estimate decision boundary
        x_decision_boundary, y_decision_boundary = fitter.compute_decision_boundary_samples(X_stacked)

        # finalize result in kernel
        kernel.finalize(fit_improvement, [kernel_dict['feature_a']]
                        , x_decision_boundary, y_decision_boundary)

        #append to list
        all_kernels.append(kernel)

    return all_kernels



################
## DUOVARIATE ##
################


def _naive_duovariate_cupy(df:pd.DataFrame
                      , search_dicts:List[dict]
                      , target_variable:str
                      , feature_space:List[str]
                      , quality_metric:str='IG_Gini') -> List[Abstract_Kernel]:
    X = df[feature_space].to_numpy(dtype=np.float64)
    Y = df[target_variable].to_numpy(dtype=np.float64).reshape((-1, 1))

    X = cp.array(X)
    Y = cp.array(Y)

    feature_name_2_index = {}
    for i in range(len(feature_space)):
        feature_name_2_index[feature_space[i]] = i

    all_kernels = []

    for kernel_dict in search_dicts:
        # slice X so that it only has the feature for the input kernel
        feature_index_a = feature_name_2_index[kernel_dict['feature_a']]
        feature_index_b = feature_name_2_index[kernel_dict['feature_b']]
        X_slice = X[:, [feature_index_a, feature_index_b]]

        # apply the kernel function to the selected feature
        kernel = get_duo_kernel(kernel_dict['kernel'])
        X_kernel = kernel.fit_and_transform(X_slice)

        # evaluate naive fit quality
        fitter = Logistic_Regression_ANN(quality_metric)
        fit_quality = fitter.compute_fit_quality(X_kernel, Y)

        # finalize result in kernel
        kernel.finalize(fit_quality, [kernel_dict['feature_a']])

        # append to list
        all_kernels.append(kernel)

    return all_kernels



def _naive_duovariate_numpy(df:pd.DataFrame
                      , search_dicts:List[dict]
                      , target_variable:str
                      , feature_space:List[str]
                      , quality_metric:str='IG_Gini') -> List[Abstract_Kernel]:
    X = df[feature_space].to_numpy(dtype=np.float64)
    Y = df[target_variable].to_numpy(dtype=np.float64).reshape((-1, 1))

    feature_name_2_index = {}
    for i in range(len(feature_space)):
        feature_name_2_index[feature_space[i]] = i

    all_kernels = []

    for kernel_dict in search_dicts:
        # slice X so that it only has the feature for the input kernel
        feature_index_a = feature_name_2_index[kernel_dict['feature_a']]
        feature_index_b = feature_name_2_index[kernel_dict['feature_b']]
        X_slice = X[:, [feature_index_a, feature_index_b]]

        # apply the kernel function to the selected feature
        kernel = get_duo_kernel(kernel_dict['kernel'])
        X_kernel = kernel.fit_and_transform(X_slice)

        # evaluate naive fit quality
        fitter = Logistic_Scikit(quality_metric)
        fit_quality = fitter.compute_fit_quality(X_kernel, Y)

        # finalize result in kernel
        kernel.finalize(fit_quality, [kernel_dict['feature_a']])

        # append to list
        all_kernels.append(kernel)

    return all_kernels



def _full_duovariate_cupy(df:pd.DataFrame
                      , search_dicts:List[dict]
                      , target_variable:str
                      , feature_space:List[str]
                      , quality_metric: str = 'IG_Gini') -> List[Abstract_Kernel]:
    X = df[feature_space].to_numpy(dtype=np.float64)
    Y = df[target_variable].to_numpy(dtype=np.float64).reshape((-1,1))

    X = cp.array(X)
    Y = cp.array(Y)

    feature_name_2_index = {}
    for i in range(len(feature_space)):
        feature_name_2_index[feature_space[i]] = i

    all_kernels = []

    #compute separability prior kernel extension
    fitter = Logistic_Regression_ANN(quality_metric)
    fit_quality_pre_kernel = fitter.compute_fit_quality(X, Y)

    for kernel_dict in search_dicts:
        #slice X so that it only has the feature for the input kernel
        feature_index_a = feature_name_2_index[kernel_dict['feature_a']]
        feature_index_b = feature_name_2_index[kernel_dict['feature_b']]
        X_slice = X[:, [feature_index_a, feature_index_b]]

        #apply the kernel function to the selected feature
        kernel = get_duo_kernel(kernel_dict['kernel'])
        X_kernel = kernel.fit_and_transform(X_slice)

        X_stacked = cp.column_stack((X, X_kernel))

        #evaluate full fit quality
        fitter = Logistic_Regression_ANN(quality_metric)
        fit_quality = fitter.compute_fit_quality(X_stacked,Y)

        fit_improvement = fit_quality - fit_quality_pre_kernel

        # estimate decision boundary
        x_decision_boundary, y_decision_boundary = fitter.compute_decision_boundary_samples(X_stacked)

        # finalize result in kernel
        kernel.finalize(fit_improvement, [kernel_dict['feature_a'], kernel_dict['feature_b']]
                        , x_decision_boundary, y_decision_boundary)

        #append to list
        all_kernels.append(kernel)

    return all_kernels




def _full_duovariate_numpy(df:pd.DataFrame
                      , search_dicts:List[dict]
                      , target_variable:str
                      , feature_space:List[str]
                      , quality_metric: str = 'IG_Gini') -> List[Abstract_Kernel]:
    X = df[feature_space].to_numpy(dtype=np.float64)
    Y = df[target_variable].to_numpy(dtype=np.float64).reshape((-1,1))

    feature_name_2_index = {}
    for i in range(len(feature_space)):
        feature_name_2_index[feature_space[i]] = i

    all_kernels = []

    #compute separability prior kernel extension
    fitter = Logistic_Scikit(quality_metric)
    fit_quality_pre_kernel = fitter.compute_fit_quality(X, Y)

    for kernel_dict in search_dicts:
        #slice X so that it only has the feature for the input kernel
        feature_index_a = feature_name_2_index[kernel_dict['feature_a']]
        feature_index_b = feature_name_2_index[kernel_dict['feature_b']]
        X_slice = X[:, [feature_index_a, feature_index_b]]

        #apply the kernel function to the selected feature
        kernel = get_duo_kernel(kernel_dict['kernel'])
        X_kernel = kernel.fit_and_transform(X_slice)

        X_stacked = np.column_stack((X, X_kernel))

        #evaluate full fit quality
        fitter = Logistic_Scikit(quality_metric)
        fit_quality = fitter.compute_fit_quality(X_stacked,Y)

        fit_improvement = fit_quality - fit_quality_pre_kernel

        #estimate decision boundary
        x_decision_boundary, y_decision_boundary = fitter.compute_decision_boundary_samples(X_stacked)

        #finalize result in kernel
        kernel.finalize(fit_improvement, [kernel_dict['feature_a'], kernel_dict['feature_b']]
                        , x_decision_boundary, y_decision_boundary)

        #append to list
        all_kernels.append(kernel)

    return all_kernels

'''

def _generic(df:pd.DataFrame
                      , search_dicts:List[dict]
                      , target_variable:str
                      , feature_space:List[str]
                      , quality_metric:str='acc'
                      , search_method:str='naive'
                      , kernel_type:str='monovariate'
                      , compute_decision_boundary:bool=False
             , use_cupy:bool=False) -> List[Abstract_Kernel]:


    X = df[feature_space].to_numpy(dtype=np.float64)
    Y = df[target_variable].to_numpy(dtype=np.float64).reshape((-1,1))

    api = np
    if use_cupy:
        X = cp.array(X)
        Y = cp.array(Y)
        api = cp
        cp.random.seed(1)

    feature_name_2_index = {}
    for i in range(len(feature_space)):
        feature_name_2_index[feature_space[i]] = i

    all_kernels = []

    # compute separability prior kernel extension
    if use_cupy:
        fitter = Logistic_Regression_ANN(quality_metric)
    else:
        fitter = Logistic_Scikit(quality_metric)
    fit_quality_pre_kernel = fitter.compute_fit_quality(X, Y)

    run_count = 0
    for kernel_dict in search_dicts:
        if run_count == 60:
            print('')

        #slice X so that it only has the feature for the input kernel
        if kernel_type == 'monovariate':
            feature_index = feature_name_2_index[kernel_dict['feature_a']]
            kernel_input_feature_names = [kernel_dict['feature_a']]
            X_slice = X[:,[feature_index]]
            # apply the kernel function to the selected feature
            kernel = get_mono_kernel(kernel_dict['kernel'], kernel_dict['standardizer'])

        if kernel_type == 'duovariate':
            feature_index_a = feature_name_2_index[kernel_dict['feature_a']]
            feature_index_b = feature_name_2_index[kernel_dict['feature_b']]
            kernel_input_feature_names = [kernel_dict['feature_a'], kernel_dict['feature_b']]
            X_slice = X[:, [feature_index_a, feature_index_b]]
            # apply the kernel function to the selected feature
            kernel = get_duo_kernel(kernel_dict['kernel'], kernel_dict['standardizer'])


        X_kernel = kernel.fit_and_transform(X_slice)

        #depending on search method, construct input space for model
        if search_method == 'naive':
            X_final = X_kernel
        if search_method == 'full':
            X_final = api.column_stack((X, X_kernel))

        #evaluate naive fit quality
        if use_cupy:
            fitter = Logistic_Regression_ANN(quality_metric)
        else:
            fitter = Logistic_Scikit(quality_metric)

        fit_quality = fitter.compute_fit_quality(X_final,Y)
        if search_method == 'full':
            fit_quality = fit_quality - fit_quality_pre_kernel

        # estimate decision boundary
        if compute_decision_boundary:
            x_decision_boundary, y_decision_boundary = fitter.compute_decision_boundary_samples(X_final)
        else:
            x_decision_boundary = None
            y_decision_boundary = None

        # finalize result in kernel
        # depending on search method, construct input space for model
        if search_method == 'naive':
            model_input_feature_names = kernel.get_kernel_feature_names(kernel_input_feature_names)
        if search_method == 'full':
            model_input_feature_names = feature_space.copy()
            for f in kernel.get_kernel_feature_names(kernel_input_feature_names):
                model_input_feature_names.append(f)

        kernel.finalize(fit_quality, kernel_input_feature_names, model_input_feature_names
                        , x_decision_boundary, y_decision_boundary)

        #append to list
        all_kernels.append(kernel)

        run_count = run_count + 1

    return all_kernels

