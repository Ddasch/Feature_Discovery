from featurediscovery.standardizers.abstract_standardizer import Abstract_Standardizer
from typing import Union

import cupy as cp
import numpy as np


SUPPORTED_STANDARDIZERS = ['dummy', 'mean', 'standard']

class Dummy_Standardizer(Abstract_Standardizer):
    
    def _fit(self,x: Union[np.ndarray, cp.ndarray]):
        pass


    def _transform(self,x: Union[np.ndarray, cp.ndarray]) -> Union[np.ndarray, cp.ndarray]:
        return x.copy()


class Mean_Centralizer(Abstract_Standardizer):

    means = None

    def _fit(self, x: Union[np.ndarray, cp.ndarray]):
        self.means = x.mean(axis=0)

    def _transform(self, x: Union[np.ndarray, cp.ndarray]) -> Union[np.ndarray, cp.ndarray]:
        x_ret = x.copy()
        x_ret = x_ret - self.means
        return x_ret



class Stand_Scaler(Abstract_Standardizer):

    means = None
    stds = None

    def _fit(self, x: Union[np.ndarray, cp.ndarray]):
        self.means = x.mean(axis=0)
        self.stds = x.std(axis=0)

    def _transform(self, x: Union[np.ndarray, cp.ndarray]) -> Union[np.ndarray, cp.ndarray]:
        x_ret = x.copy()
        x_ret = x_ret - self.means
        x_ret = x_ret / self.stds
        return x_ret

