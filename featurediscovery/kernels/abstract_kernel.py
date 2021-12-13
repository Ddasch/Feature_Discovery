from abc import ABC, abstractmethod
from typing import Union

import cupy as cp
import numpy as np

from featurediscovery.standardizers.standardizers import *

class Abstract_Kernel(ABC):
    standardizer = None
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




