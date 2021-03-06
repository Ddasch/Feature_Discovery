from abc import ABC, abstractmethod
from typing import Union, List

import cupy as cp
import numpy as np
import pandas as pd

from featurediscovery.standardizers.standardizers import get_scaler
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

from featurediscovery.standardizers.abstract_standardizer import Abstract_Standardizer


class Abstract_Kernel(ABC):
    standardizer:Abstract_Standardizer = None
    finalized:bool = False
    kernel_quality:float = None
    api:str = None
    kernel_input_features:List[str] = None
    model_input_features: List[str] = None
    x_decision_boundary: Union[np.ndarray, cp.ndarray] = None
    y_decision_boundary: Union[np.ndarray, cp.ndarray] = None

    def __init__(self, standardizer:str=None):
        self.standardizer = get_scaler(standardizer)


    def fit(self, x: Union[np.ndarray, cp.ndarray]):

        if len(x.shape) != 2:
            raise Exception('x must be 2 dimensional, first dimension indicating the sample index and second the feature index')

        if type(x) == cp.ndarray:
            self.api = 'cupy'
        else:
            self.api = 'numpy'

        x_std = self.standardizer.fit_and_transform(x)

        self._fit(x_std)

    def transform(self, x: Union[np.ndarray, cp.ndarray]) -> Union[np.ndarray, cp.ndarray]:
        if len(x.shape) != 2:
            raise Exception('x must be 2 dimensional, first dimension indicating the sample index and second the feature index')

        x_std = self.standardizer.transform(x)
        return self._transform(x_std)

    def fit_and_transform(self, x: Union[np.ndarray, cp.ndarray]) -> Union[np.ndarray, cp.ndarray]:
        #x_std = self.standardizer.fit_and_transform(x)

        self.fit(x)
        return self.transform(x)

    @abstractmethod
    def _fit(self, x: Union[np.ndarray, cp.ndarray]):
        pass

    @abstractmethod
    def _transform(self, x: Union[np.ndarray, cp.ndarray]) -> Union[np.ndarray, cp.ndarray]:
        pass

    def finalize(self, quality:float, kernel_input_features:List[str], model_input_features:List[str]
                 , x_decision_boundary: Union[np.ndarray, cp.ndarray]
                 , y_decision_boundary: Union[np.ndarray, cp.ndarray]):
        '''
        Once kernel quality has been computed, store meta-information about this kernel
        :param quality: the quality metric for sorting the kernel by quality
        :param kernel_input_features: feature names so that in the future the kernel can be directly applied on the dataframe
        :param x_decision_boundary
        :param y_decision_boundary
        :return:
        '''
        self.finalized = True
        self.kernel_quality = quality
        self.kernel_input_features = kernel_input_features
        self.model_input_features = model_input_features


        '''
        #NOTE: raw input feature are not standardized when also given to the model, only when computing kernel
                transformation. Inverse transform not needed
        #if the model feature space alse included the raw features that were the input to the kernel (aka full search),
        # inverse transform the decision boundary back to its original space
        model_space_kernel_feature_indexi = []
        for f in self.kernel_input_features:
            if f in self.model_input_features:
                model_space_kernel_feature_indexi.append(model_input_features.index(f))

        model_space_non_kernel_feature_indexi = [i for i in range(len(self.model_input_features))
                                     if self.model_input_features[i] not in self.kernel_input_features]

        if len(model_space_kernel_feature_indexi) > 0:
            x_decision_boundary_kernel_input = x_decision_boundary[:,model_space_kernel_feature_indexi]
            x_decision_boundary_kernel_input_inverse_transformed = self.standardizer.inverse_transform(x_decision_boundary_kernel_input)

            x_decision_boundary[:,model_space_kernel_feature_indexi] = x_decision_boundary_kernel_input_inverse_transformed
            #for i in range(len(model_space_kernel_feature_indexi)):
            #    x_decision_boundary[:,model_space_kernel_feature_indexi[i]] = x_decision_boundary_kernel_input_inverse_transformed[i]
        '''

        self.x_decision_boundary = x_decision_boundary
        self.y_decision_boundary = y_decision_boundary

    def get_decision_boundary(self, feature_names:List[str]):

        if not self.finalized:
            raise Exception('kernel not finalized yet')

        x_indexi = []
        for f in feature_names:
            if f in self.kernel_input_features:
                x_indexi.append(self.kernel_input_features.index(f))
                continue

            if f in self.get_kernel_feature_names():
                x_indexi.append(self.get_kernel_feature_names().index(f))

        return self.x_decision_boundary[:,x_indexi]


    def apply(self, df:pd.DataFrame):

        if not self.finalized:
            raise Exception('Cannot apply kernel to DF. Kernel is not finalized yet')

        if len(self.kernel_input_features) == 0 or self.kernel_input_features is None:
            raise Exception('Illegal kernel state. Kernel is tagged as finalized but has no feature name list...')

        for f in self.kernel_input_features:
            if f not in df.columns:
                raise Exception('Cannot apply kernel. Feature {} not in dataframe'.format(f))



        X = df[self.kernel_input_features].to_numpy(dtype=np.float64)
        if self.api == 'cupy':
            X = cp.array(X)

        X_kernel = self.transform(X)

        if self.api == 'cupy':
            X_kernel = X_kernel.get()

        kernel_feature_names = self.get_kernel_feature_names()
        if X_kernel.shape[1] != len(kernel_feature_names):
            raise Exception('Kernel is returning unequal amount of feature names than features')

        for i in range(X_kernel.shape[1]):
            df[kernel_feature_names[i]] = X_kernel[:,i]

        return df

    @abstractmethod
    def get_kernel_name(self) -> str:
        pass

    @abstractmethod
    def get_kernel_feature_names(self, input_features:List[str]=None) -> List[str]:
        pass

    @abstractmethod
    def get_kernel_type(self) -> str:
        pass
