from abc import ABC, abstractmethod
from typing import Union, List

import cupy as cp
import numpy as np
import pandas as pd

from featurediscovery.kernels.abstract_kernel import Abstract_Kernel
from featurediscovery.standardizers.standardizers import *

class Abstract_Monovariate_Kernel(Abstract_Kernel):
    standardizer = None
    def __init__(self, standardizer:str=None):
        super().__init__(standardizer)



    def fit(self, x: Union[np.ndarray, cp.ndarray]):
        super().fit(x)

        if x.shape[1] > 1:
            print('WARNING - Executing monovariate kernel on multiple features. Applying kernel on every feature separately..' )

        x_std = self.standardizer.fit_and_transform(x)

        self._fit(x_std)

    def transform(self, x: Union[np.ndarray, cp.ndarray], suppres_warning:bool=False) -> np.ndarray:
        super().transform(x)

        if x.shape[1] > 1 and not suppres_warning:
            print('WARNING - Executing monovariate kernel on multiple features. Applying kernel on every feature separately..' )

        x_std = self.standardizer.transform(x)

        return self._transform(x_std)


    @abstractmethod
    def _fit(self, x: Union[np.ndarray, cp.ndarray]):
        pass

    @abstractmethod
    def _transform(self, x: Union[np.ndarray, cp.ndarray]) -> Union[np.ndarray, cp.ndarray]:
        pass

    def finalize(self, quality:float, features:List[str]):
        super().finalize(quality,features)

        self.kernel_quality = quality
        self.features = features

    def apply(self, df:pd.DataFrame):
        if not self.finalized:
            raise Exception('Attempting to apply kernel that has not been finalized yet')

        #TODO apply kernel and return df



