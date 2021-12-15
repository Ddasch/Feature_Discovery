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

    def __init__(self, standardizer:str=None):

        if standardizer is not None and standardizer not in SUPPORTED_STANDARDIZERS:
            raise Exception('Unsupported standardizer: {}'.format(standardizer))

        if standardizer in ['Dummy', 'dummy', 'none', 'None'] or standardizer is None:
            self.standardizer = Dummy_Standardizer()

        if standardizer in ['Mean', 'mean']:
            self.standardizer = Mean_Centralizer()

        if standardizer in ['standard']:
            self.standardizer = Stand_Scaler()


    @abstractmethod
    def fit(self, x: Union[np.ndarray, cp.ndarray]):

        if len(x.shape) != 2:
            raise Exception('x must be 2 dimensional, first dimension indicating the sample index and second the feature index')


    @abstractmethod
    def transform(self, x: Union[np.ndarray, cp.ndarray]) -> cp.ndarray:
        if len(x.shape) != 2:
            raise Exception('x must be 2 dimensional, first dimension indicating the sample index and second the feature index')



    def fit_and_transform(self, x: Union[np.ndarray, cp.ndarray]) -> cp.ndarray:
        x_std = self.standardizer.fit_and_transform(x)

        self.fit(x_std)
        return self.transform(x_std)


    @abstractmethod
    def _fit(self, x: Union[np.ndarray, cp.ndarray]):
        pass

    @abstractmethod
    def _transform(self, x: Union[np.ndarray, cp.ndarray]) -> Union[np.ndarray, cp.ndarray]:
        pass

    @abstractmethod
    def finalize(self, quality:float, features:List[str]):
        '''
        Once kernel quality has been computed, store meta-information about this kernel
        :param quality: the quality metric for sorting the kernel by quality
        :param features: feature names so that in the future the kernel can be directly applied on the dataframe
        :return:
        '''
        self.finalized=True

        pass

    @abstractmethod
    def apply(self, df:pd.DataFrame):
        pass

    @abstractmethod
    def get_kernel_name(self) -> str:
        pass
