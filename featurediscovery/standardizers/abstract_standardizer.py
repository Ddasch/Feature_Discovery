from abc import ABC, abstractmethod
from typing import Union

import cupy as cp
import numpy as np


class Abstract_Standardizer(ABC):

    def __init__(self):
        pass

    def fit(self, x: Union[np.ndarray, cp.ndarray]):

        if len(x.shape) != 2:
            raise Exception('x must be 2 dimensional, first dimension indicating the sample index and second the feature index')
        self._fit(x)

    def transform(self, x: Union[np.ndarray, cp.ndarray]) -> np.ndarray:
        if len(x.shape) != 2:
            raise Exception('x must be 2 dimensional, first dimension indicating the sample index and second the feature index')
        return self._transform(x)

    def fit_and_transform(self, x: Union[np.ndarray, cp.ndarray]) -> np.ndarray:
        if len(x.shape) != 2:
            raise Exception('x must be 2 dimensional, first dimension indicating the sample index and second the feature index')

        self.fit(x)
        return self.transform(x)


    @abstractmethod
    def _fit(self, x: Union[np.ndarray, cp.ndarray]):
        pass

    @abstractmethod
    def _transform(self, x: Union[np.ndarray, cp.ndarray]) -> Union[np.ndarray, cp.ndarray]:
        pass

    @abstractmethod
    def get_standardizer_name(self) -> str:
        pass
