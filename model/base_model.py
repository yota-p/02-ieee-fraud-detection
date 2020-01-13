from abc import ABCMeta, abstractmethod
import pandas as pd

from util.mylog import timer


class BaseModel(metaclass=ABCMeta):
    '''
    Class to wrap models.
    Contain specific models from frameworks at self.core.
    '''
    core = None

    @abstractmethod
    def train(self, X_train, y_train):
        raise NotImplementedError

    @abstractmethod
    def train_and_validate(X_train, y_train, X_val, y_val):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X_test):
        raise NotImplementedError

    @timer
    def save(self, path_to_save):
        self.core.to_pickle(path_to_save)

    @timer
    def load(self, path_to_load):
        self.core = pd.read_pickle(path_to_load)
