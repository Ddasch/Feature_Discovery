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

import cupy as cp
import numpy as np
from typing import Union

from sklearn.linear_model import RidgeClassifier, LogisticRegression

from featurediscovery.fitter.abstract_fitter import Abstract_Fitter



class Linear_Scikit(Abstract_Fitter):

    model:RidgeClassifier = None

    def _fit(self, x: Union[np.ndarray, cp.ndarray], y:Union[np.ndarray, cp.ndarray]):
        if type(x) != np.ndarray or type(y) != np.ndarray:
            raise Exception('Scikit fitters require np.ndarray as input. Received {} instead'.format(type(x)))

        ridge = RidgeClassifier()

        ridge.fit(x,y.reshape(-1))

        self.model = ridge

    def _score(self, x: Union[np.ndarray, cp.ndarray]):
        if type(x) != np.ndarray:
            raise Exception('Scikit fitters require np.ndarray as input. Received {} instead'.format(type(x)))

        return self.model.predict(x).reshape(-1,1)


class Logistic_Scikit(Abstract_Fitter):

    model:LogisticRegression = None

    def _fit(self, x: Union[np.ndarray, cp.ndarray], y:Union[np.ndarray, cp.ndarray], debug:bool=False):
        if type(x) != np.ndarray or type(y) != np.ndarray:
            raise Exception('Scikit fitters require np.ndarray as input. Received {} instead'.format(type(x)))

        logistic = LogisticRegression(solver='liblinear')

        logistic.fit(x,y.reshape(-1))

        self.model = logistic

    def _score(self, x: Union[np.ndarray, cp.ndarray]):
        if type(x) != np.ndarray:
            raise Exception('Scikit fitters require np.ndarray as input. Received {} instead'.format(type(x)))

        return self.model.predict(x).reshape(-1,1)

    def _score_prob(self, x: Union[np.ndarray, cp.ndarray]):
        probas = self.model.predict_proba(x)
        return probas[:,1].reshape(-1,1)