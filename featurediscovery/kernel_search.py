import cupy as cp
import numpy as np
import pandas as pd

from typing import List

from featurediscovery.kernels.abstract_kernel import Abstract_Kernel
from featurediscovery.kernels.monovariate.monovariate_kernels import SUPPORTED_MONOVARIATE_KERNELS
from featurediscovery.kernels.duovariate.duovariate_kernels import SUPPORTED_DUOVARIATE_KERNELS
from featurediscovery.standardizers.standardizers import SUPPORTED_STANDARDIZERS
from featurediscovery.util import general_utilities
from featurediscovery.util.exceptions import *
from featurediscovery.search_routines import naive_monovariate, full_monovariate, naive_duovariate, full_duovariate
from featurediscovery import plotter
from featurediscovery import exporter

def _search(df:pd.DataFrame
            , target_variable:str
            , feature_space:List[str]
            , eval_method:str = 'full'
            , mandatory_features:List[str] = None
            , monovariate_kernels: List[str] = None
            , duovariate_kernels: List[str] = None
            , feature_standardizers:List[str] = None
            , use_cupy:str = 'auto'
            ) -> List[Abstract_Kernel]:

    #check df columns
    if type(target_variable) != str:
        raise SetupException('Target variable must be of type str. instead it is {}'.format(type(target_variable)))

    if target_variable not in df.columns:
        raise SetupException('Target variable {} not in df'.format(target_variable))

    for f in feature_space:
        if f not in df.columns:
            raise SetupException('Feature {} not in df'.format(f))


    #check that all inputs are numeric
    datatypes = df.dtypes
    for f in feature_space:
        if datatypes[f] not in [float, int, np.float32, np.float64, np.int32, np.int64]:
            raise SetupException('Feature {} is not numeric. Must be in [float, int, np.float32, np.float64, np.int32, np.int64]'.format(f))

    #check that mandatory feature is in feature space
    if mandatory_features is not None and len(mandatory_features) > 0:
        for m in mandatory_features:
            if m not in feature_space:
                raise SetupException('mandatory feature {} not in feature space'.format(m))

    #check that target column is one-hot encoded
    if datatypes[target_variable] in [int,  np.int32, np.int64]:
        df[target_variable] = df[target_variable].astype(np.float64)
        datatypes = df.dtypes

    if len(set(df[target_variable].unique()) - {0.0, 1.0}) > 0:
        raise SetupException('Target variable {} not one-hot encoded. Must have values of {{0.0, 1.0/}}, but instead has {}'.format(target_variable, set(df[target_variable].unique())))

    if monovariate_kernels is None:
        monovariate_kernels = []

    if duovariate_kernels is None:
        duovariate_kernels = []

    #check kernel list specification
    if (monovariate_kernels is None or len(monovariate_kernels) == 0)\
            and (duovariate_kernels is None or len(duovariate_kernels) == 0):
        raise SetupException('Must supply at least one kernel type to try')

    for mono in monovariate_kernels:
        if mono not in SUPPORTED_MONOVARIATE_KERNELS:
            raise SetupException('Listed monovariate kernel {} not supported. Supported kernels are: {}'.format(mono, SUPPORTED_MONOVARIATE_KERNELS))

    for duo in duovariate_kernels:
        if duo not in SUPPORTED_DUOVARIATE_KERNELS:
            raise SetupException('Listed duovariate kernel {} not supported. Supported kernels are: {}'.format(duovariate_kernels, SUPPORTED_DUOVARIATE_KERNELS))

    #check that eval method correctly specified
    if eval_method not in ['full', 'naive']:
        raise SetupException('Eval method must be in [full, naive]')

    #check that standardizer lise is correct
    if feature_standardizers is None:
        raise SetupException('No standardizer list provided. Must supply at least 1 out of {}'.format(SUPPORTED_STANDARDIZERS))

    for s in feature_standardizers:
        if s not in SUPPORTED_STANDARDIZERS:
            raise SetupException('Standardizer {} not supported. Supported standardizers: {}'.format(s, SUPPORTED_STANDARDIZERS))


    if use_cupy not in ['yes', 'no', 'auto']:
        raise SetupException('Unsupported cupy use mode {}. Must be in [yes, no, auto]'.format(use_cupy))


    X = df[feature_space].to_numpy(dtype=np.float32)
    #y = df[target_variable].to_numpy(dtype=np.float32)

    #based on provided bool (or lack thereof), detimine usage of cupy
    if use_cupy is None:
        use_cupy = 'auto'

    if use_cupy == 'yes':
        use_cupy = True
    if use_cupy == 'no':
        use_cupy = False
    if use_cupy == 'auto':
        cupy_thresh = 1000000 #TODO benchmark
        use_cupy = X.shape[0]*X.shape[1] > cupy_thresh


    #generate list of monovariate kernels to try
    features_to_try = feature_space
    if mandatory_features is not None:
        features_to_try = mandatory_features
    mono_kernel_dicts = general_utilities._generate_all_list_combinations(kernel=monovariate_kernels
                                                                          , feature_a=features_to_try
                                                                          , standardizer=feature_standardizers)

    #generate list of duovariate kernels to try
    duo_kernel_dicts = general_utilities.create_duovariate_combination_dicts(mandatory_features=mandatory_features
                                                                             , feature_space=feature_space
                                                                             , kernels=duovariate_kernels
                                                                             , standardizer=feature_standardizers)


    all_kernels = []
    if len(mono_kernel_dicts) > 0:
        if eval_method == 'naive':
            mono_results = naive_monovariate(df=df, search_dicts=mono_kernel_dicts, feature_space=feature_space
                                             , target_variable=target_variable, use_cupy=use_cupy)
        else:
            mono_results = full_monovariate(df=df, search_dicts=mono_kernel_dicts, feature_space=feature_space
                                            , target_variable=target_variable, use_cupy=use_cupy)

        for kernel in mono_results:
            all_kernels.append(kernel)

    if len(duo_kernel_dicts) > 0:
        if eval_method == 'naive':
            duo_results = naive_duovariate(df=df, search_dicts=duo_kernel_dicts, feature_space=feature_space
                                             , target_variable=target_variable, use_cupy=use_cupy)
        else:
            duo_results = full_duovariate(df=df, search_dicts=duo_kernel_dicts, feature_space=feature_space
                                           , target_variable=target_variable, use_cupy=use_cupy)

        for kernel in duo_results:
            all_kernels.append(kernel)

    return all_kernels




def evaluate_kernels(df:pd.DataFrame
            , target_variable: str
            , feature_space: List[str]
            , eval_method: str = 'full'
            , mandatory_features: List[str] = None
            , monovariate_kernels: List[str] = None
            , duovariate_kernels: List[str] = None
            , feature_standardizers:List[str] = None
            , use_cupy: str = 'auto'
            , plot_ranking:bool=True
            , plot_individual_kernels:bool=False
            , kernel_plot_mode: str = 'scree'
            , export_folder: str = None
            , export_ranking: bool = False
            , export_formats: List[str] = None
            , export_individual_kernel_plots: bool = False
         ) -> List[Abstract_Kernel]:

    kernel_list = _search(df=df
                          , target_variable=target_variable
                          , feature_space=feature_space
                          , eval_method=eval_method
                          , mandatory_features=mandatory_features
                          , monovariate_kernels=monovariate_kernels
                          , duovariate_kernels=duovariate_kernels
                          , feature_standardizers=feature_standardizers
                          , use_cupy=use_cupy)


    kernel_list.sort(key=lambda x: x.kernel_quality, reverse=True)

    if plot_individual_kernels or export_individual_kernel_plots:
        for k in kernel_list:
            plotter.plot_kernel(df=df, kernel=k, target_variable=target_variable
                        , mode=kernel_plot_mode
                        , to_screen=plot_individual_kernels
                        , to_file=export_individual_kernel_plots
                        , export_folder=export_folder)


    if plot_ranking or (export_ranking and 'png' in export_formats):
        plotter.plot_ranking(kernel_list
                     , to_screen=plot_ranking
                     , to_file=export_ranking
                     , export_folder=export_folder
                     )

    if export_formats is not None:
        for f in export_formats:
            if f in ['csv', 'json']:
                exporter.export_kernel_ranking(kernel_list, export_folder, f)

    return kernel_list
