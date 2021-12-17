from featurediscovery.standardizers.abstract_standardizer import Abstract_Standardizer
from typing import Union

import cupy as cp
import numpy as np


SUPPORTED_STANDARDIZERS = ['dummy', 'mean', 'standard', 'raw']


def get_scaler(scaler_name:str) -> Abstract_Standardizer:

    if scaler_name is not None and scaler_name not in SUPPORTED_STANDARDIZERS:
        raise Exception('Unsupported standardizer: {}'.format(scaler_name))

    if scaler_name in ['Dummy', 'dummy', 'none', 'None', 'raw'] or scaler_name is None:
        return Dummy_Standardizer()

    if scaler_name in ['Mean', 'mean']:
        return Mean_Centralizer()

    if scaler_name in ['standard']:
        return Stand_Scaler()



class Dummy_Standardizer(Abstract_Standardizer):
    
    def _fit(self,x: Union[np.ndarray, cp.ndarray]):
        pass


    def _transform(self,x: Union[np.ndarray, cp.ndarray]) -> Union[np.ndarray, cp.ndarray]:
        return x.copy()

    def get_standardizer_name(self):
        return 'raw'

class Mean_Centralizer(Abstract_Standardizer):

    means = None

    def _fit(self, x: Union[np.ndarray, cp.ndarray]):
        self.means = x.mean(axis=0)

    def _transform(self, x: Union[np.ndarray, cp.ndarray]) -> Union[np.ndarray, cp.ndarray]:
        x_ret = x.copy()
        x_ret = x_ret - self.means
        return x_ret

    def get_standardizer_name(self):
        return 'centralized'

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

    def get_standardizer_name(self):
        return 'std-scaled'