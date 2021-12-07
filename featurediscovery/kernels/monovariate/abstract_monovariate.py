from abc import ABC, abstractmethod



class Abstract_Monovariate_Kernel(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def transform(self):
        pass

    @abstractmethod
    def fit_and_transform(self):
        pass




