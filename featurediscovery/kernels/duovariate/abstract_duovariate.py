from abc import ABC, abstractmethod
from typing import Union

import cupy as cp
import numpy as np

from featurediscovery.kernels.abstract_kernel import Abstract_Kernel
from featurediscovery.standardizers.standardizers import *

class Abstract_Duovariate_Kernel(Abstract_Kernel):
    standardizer = None
    def __init__(self, standardizer:str=None):
        super().__init__(standardizer)



    def fit(self, x: Union[np.ndarray, cp.ndarray]):
        super().fit(x)

        if x.shape[1] != 2:
            raise Exception('Amount of input columns must be exactly 2')

        x_std = self.standardizer.fit_and_transform(x)

        self._fit(x_std)

    def transform(self, x: Union[np.ndarray, cp.ndarray]) -> np.ndarray:
        super().transform(x)

        if x.shape[1] != 2:
            raise Exception('Amount of input columns must be exactly 2')

        x_std = self.standardizer.transform(x)

        return self._transform(x_std)


    @abstractmethod
    def _fit(self, x: Union[np.ndarray, cp.ndarray]):
        pass

    @abstractmethod
    def _transform(self, x: Union[np.ndarray, cp.ndarray]) -> Union[np.ndarray, cp.ndarray]:
        pass




