

import numpy as np
import cupy as cp
import pandas as pd
from typing import List

from featurediscovery.kernels.abstract_kernel import Abstract_Kernel
from featurediscovery.fitter.cupy_fitter import Logistic_Regression_ANN
from featurediscovery.fitter.sklearn_fitters import LogisticRegression
from featurediscovery.kernels.monovariate.monovariate_kernels import get_mono_kernel


def naive_monovariate(df:pd.DataFrame
                      , search_dicts:List[dict]
                      , target_variable:str
                      , feature_space:List[str]
                      , use_cupy:bool=True) -> List[Abstract_Kernel]:

    if use_cupy:
        return _naive_monovariate_cupy(df, search_dicts, target_variable=target_variable, feature_space=feature_space)
    else:
        return _naive_monovariate_numpy(df, search_dicts, target_variable=target_variable, feature_space=feature_space)


def full_monovariate(df:pd.DataFrame
                      , search_dicts:List[dict]
                      , target_variable:str
                      , feature_space:List[str]
                      , use_cupy:bool=True) -> List[Abstract_Kernel]:

    if use_cupy:
        pass
        #TODO
    else:
        pass
        #TODO


def _naive_monovariate_cupy(df:pd.DataFrame
                      , search_dicts:List[dict]
                      , target_variable:str
                      , feature_space:List[str]) -> List[Abstract_Kernel]:


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
        fitter = Logistic_Regression_ANN('gini')
        fit_quality = fitter.compute_fit_quality(X_kernel,Y)

        #finalize result in kernel
        kernel.finalize(fit_quality, [kernel_dict['feature_a']])

        #append to list
        all_kernels.append(kernel)

    return all_kernels





def _naive_monovariate_numpy(df:pd.DataFrame
                      , search_dicts:List[dict]
                      , target_variable:str
                      , feature_space:List[str]) -> List[Abstract_Kernel]:

    pass