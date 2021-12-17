from abc import ABC, abstractmethod
from typing import Union, List

import cupy as cp
import numpy as np
import pandas as pd

from featurediscovery.kernels.abstract_kernel import Abstract_Kernel


class Abstract_Duovariate_Kernel(Abstract_Kernel):

    def fit(self, x: Union[np.ndarray, cp.ndarray]):
        if x.shape[1] != 2:
            raise Exception('Amount of input columns must be exactly 2')

        super().fit(x)

    def transform(self, x: Union[np.ndarray, cp.ndarray]) -> Union[np.ndarray, cp.ndarray]:
        if x.shape[1] != 2:
            raise Exception('Amount of input columns must be exactly 2')

        return super().transform(x)

    def finalize(self, quality: float, features: List[str]):
        if len(features) != 2:
            raise Exception('Duovariate kernel requires exactly 2 feature names. Only the following names were provided {}'.format(features))

        super().finalize(quality, features)



