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



class Fit_Quality_Metric(ABC):

    @abstractmethod
    def score_fit_quality(self, y: Union[np.ndarray, cp.ndarray]
                          , y_hat: Union[np.ndarray, cp.ndarray]) -> float:
        pass



