import cupy as cp
import numpy as np
from typing import Union

from sklearn.linear_model import RidgeClassifier

from featurediscovery.fitter.abstract_fitter import Abstract_Fitter



class Linear_Scikit(Abstract_Fitter):

    model:RidgeClassifier = None

    def _fit(self, x: Union[np.ndarray, cp.ndarray], y:Union[np.ndarray, cp.ndarray]):
        ridge = RidgeClassifier()

        ridge.fit(x.get(),y.get())

        self.model = ridge

    def _score(self, x: Union[np.ndarray, cp.ndarray]):
        return cp.array(self.model.predict(x.get()).reshape(-1,1))