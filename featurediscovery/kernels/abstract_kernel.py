from abc import ABC, abstractmethod
from typing import Union, List

import cupy as cp
import numpy as np
import pandas as pd

from featurediscovery.standardizers.standardizers import *
from featurediscovery.standardizers.abstract_standardizer import Abstract_Standardizer


class Abstract_Kernel(ABC):
    standardizer:Abstract_Standardizer = None
    finalized:bool = False
    kernel_quality:float = None
    api:str = None
    features:List[str] = None

    def __init__(self, standardizer:str=None):

        if standardizer is not None and standardizer not in SUPPORTED_STANDARDIZERS:
            raise Exception('Unsupported standardizer: {}'.format(standardizer))

        if standardizer in ['Dummy', 'dummy', 'none', 'None'] or standardizer is None:
            self.standardizer = Dummy_Standardizer()

        if standardizer in ['Mean', 'mean']:
            self.standardizer = Mean_Centralizer()

        if standardizer in ['standard']:
            self.standardizer = Stand_Scaler()

    def fit(self, x: Union[np.ndarray, cp.ndarray]):

        if len(x.shape) != 2:
            raise Exception('x must be 2 dimensional, first dimension indicating the sample index and second the feature index')

        if type(x) == cp.ndarray:
            self.api = 'cupy'
        else:
            self.api = 'numpy'

        x_std = self.standardizer.fit_and_transform(x)

        self._fit(x_std)

    def transform(self, x: Union[np.ndarray, cp.ndarray]) -> Union[np.ndarray, cp.ndarray]:
        if len(x.shape) != 2:
            raise Exception('x must be 2 dimensional, first dimension indicating the sample index and second the feature index')

        x_std = self.standardizer.transform(x)
        return self._transform(x_std)

    def fit_and_transform(self, x: Union[np.ndarray, cp.ndarray]) -> Union[np.ndarray, cp.ndarray]:
        #x_std = self.standardizer.fit_and_transform(x)

        self.fit(x)
        return self.transform(x)

    @abstractmethod
    def _fit(self, x: Union[np.ndarray, cp.ndarray]):
        pass

    @abstractmethod
    def _transform(self, x: Union[np.ndarray, cp.ndarray]) -> Union[np.ndarray, cp.ndarray]:
        pass

    def finalize(self, quality:float, features:List[str]):
        '''
        Once kernel quality has been computed, store meta-information about this kernel
        :param quality: the quality metric for sorting the kernel by quality
        :param features: feature names so that in the future the kernel can be directly applied on the dataframe
        :return:
        '''
        self.finalized = True
        self.kernel_quality = quality
        self.features = features

    def apply(self, df:pd.DataFrame):

        if not self.finalized:
            raise Exception('Cannot apply kernel to DF. Kernel is not finalized yet')

        if len(self.features) == 0 or self.features is None:
            raise Exception('Illegal kernel state. Kernel is tagged as finalized but has no feature name list...')

        for f in self.features:
            if f not in df.columns:
                raise Exception('Cannot apply kernel. Feature {} not in dataframe'.format(f))



        X = df[self.features].to_numpy(dtype=np.float64)
        if self.api == 'cupy':
            X = cp.array(X)

        X_kernel = self.transform(X)

        if self.api == 'cupy':
            X_kernel = X_kernel.get()

        kernel_feature_names = self.get_kernel_feature_names()
        if X_kernel.shape[1] != len(kernel_feature_names):
            raise Exception('Kernel is returning unequal amount of feature names than features')

        for i in range(X_kernel.shape[1]):
            df[kernel_feature_names[i]] = X_kernel[:,i]

        return df

    @abstractmethod
    def get_kernel_name(self) -> str:
        pass

    @abstractmethod
    def get_kernel_feature_names(self) -> List[str]:
        pass
