from abc import ABC, abstractmethod
from typing import Union, List

import cupy as cp
import numpy as np

from featurediscovery.kernels.abstract_kernel import Abstract_Kernel

class Abstract_Monovariate_Kernel(Abstract_Kernel):

    def fit(self, x: Union[np.ndarray, cp.ndarray]):
        if x.shape[1] > 1:
            print('WARNING - Executing monovariate kernel on multiple features. Applying kernel on every feature separately..' )
        super().fit(x)

    def transform(self, x: Union[np.ndarray, cp.ndarray], suppres_warning:bool=False) -> np.ndarray:
        if x.shape[1] > 1 and not suppres_warning:
            print('WARNING - Executing monovariate kernel on multiple features. Applying kernel on every feature separately..' )
        return super().transform(x)

    def finalize(self, quality:float, kernel_input_features:List[str], model_input_features:List[str]
                 , x_decision_boundary: Union[np.ndarray, cp.ndarray]
                 , y_decision_boundary: Union[np.ndarray, cp.ndarray]):
        if len(kernel_input_features) != 1:
            raise Exception('Monovariate kernel requires exactly 1 feature name. The following names were provided {}'.format(kernel_input_features))

        super().finalize(quality, kernel_input_features, model_input_features, x_decision_boundary, y_decision_boundary)





