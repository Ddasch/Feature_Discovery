from abc import ABC, abstractmethod
from typing import Union, List

import cupy as cp
import numpy as np
import pandas as pd

from featurediscovery.kernels.abstract_kernel import Abstract_Kernel
from featurediscovery.standardizers.standardizers import *

class Abstract_Monovariate_Kernel(Abstract_Kernel):

    def fit(self, x: Union[np.ndarray, cp.ndarray]):
        if x.shape[1] > 1:
            print('WARNING - Executing monovariate kernel on multiple features. Applying kernel on every feature separately..' )
        super().fit(x)

    def transform(self, x: Union[np.ndarray, cp.ndarray], suppres_warning:bool=False) -> np.ndarray:
        if x.shape[1] > 1 and not suppres_warning:
            print('WARNING - Executing monovariate kernel on multiple features. Applying kernel on every feature separately..' )
        return super().transform(x)





