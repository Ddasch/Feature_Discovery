
from abc import ABC, abstractmethod
from typing import Union, List

import cupy as cp
import numpy as np



class Fit_Quality_Metric(ABC):

    @abstractmethod
    def score_fit_quality(self, y: Union[np.ndarray, cp.ndarray]
                          , y_hat: Union[np.ndarray, cp.ndarray]) -> float:
        pass



