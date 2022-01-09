# Copyright 2021-2022 by Frederik Christiaan Schadd
# All rights reserved
#
# Licensed under the GNU Lesser General Public License version 2.1 ;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://opensource.org/licenses/LGPL-2.1
#
# or consult the LICENSE file included in the project.

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

    def transform(self, x: Union[np.ndarray, cp.ndarray], suppres_warning:bool=False) -> Union[np.ndarray, cp.ndarray]:
        if x.shape[1] > 1 and not suppres_warning:
            print('WARNING - Executing monovariate kernel on multiple features. Applying kernel on every feature separately..' )

        x_ret = super().transform(x)
        assert x_ret.shape[0] == x.shape[0]
        assert x_ret.shape[1] == x.shape[1]
        return x_ret

    def finalize(self, quality:float, kernel_input_features:List[str], model_input_features:List[str]
                 , x_decision_boundary: Union[np.ndarray, cp.ndarray]
                 , y_decision_boundary: Union[np.ndarray, cp.ndarray]):
        if len(kernel_input_features) != 1:
            raise Exception('Monovariate kernel requires exactly 1 feature name. The following names were provided {}'.format(kernel_input_features))

        super().finalize(quality, kernel_input_features, model_input_features, x_decision_boundary, y_decision_boundary)


    def get_kernel_feature_names(self, input_features: List[str] = None):
        if input_features is None:
            if not self.finalized:
                raise Exception(
                    'Cannot determine output feature name list of unfinalized kernel without supplying an input name list')
            f1 = self.kernel_input_features[0]

        else:
            if len(input_features) != 1:
                raise Exception('Input feature name list must be of length 2 for duovariate kernels')

            f1 = input_features[0]

        return self._get_kernel_feature_names(f1)

    @abstractmethod
    def _get_kernel_feature_names(self, f1: str):
        pass


